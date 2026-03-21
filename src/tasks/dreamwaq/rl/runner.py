import os
import time
from typing import Any, Dict, Optional

import torch
import wandb
from tensordict import TensorDict

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
    attach_metadata_to_onnx,
    get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner

from ..cenet import CENet, CenetRolloutStorage


class DreamWaqOnPolicyRunner(MjlabOnPolicyRunner):
    """On-policy runner for DreamWaq algorithm with CENet."""

    env: RslRlVecEnvWrapper

    def __init__(
        self,
        env: RslRlVecEnvWrapper,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        """Initialize DreamWaq runner with CENet."""
        # Call parent constructor first (sets up algorithm, logger, etc.)
        super().__init__(env, train_cfg, log_dir, device)

        # Extract DreamWaq-specific hyperparameters from train_cfg
        self.history_length = train_cfg.get("history_length", 5)
        self.cenet_beta = train_cfg.get("cenet_beta", 1.0)
        self.cenet_beta_limit = train_cfg.get("cenet_beta_limit", 4.0)
        self.cenet_learning_rate = train_cfg.get("cenet_learning_rate", 0.001)
        self.num_context = train_cfg.get("num_context", 16)
        self.num_estvel = train_cfg.get("num_estvel", 3)
        self.raw_obs_dim = train_cfg.get("raw_obs_dim", 45)
        self.cenet_beta_anneal_steps = train_cfg.get("cenet_beta_anneal_steps", 0)
        self.cenet_update_freq = train_cfg.get("cenet_update_freq", 1)
        self.cenet_hidden_dims = tuple(train_cfg.get("cenet_hidden_dims", (256, 128)))

        # Initialize CENet
        input_dim = self.history_length * self.raw_obs_dim
        self.cenet = CENet(
            input_dim=input_dim,
            beta=self.cenet_beta,
            beta_limit=self.cenet_beta_limit,
            learning_rate=self.cenet_learning_rate,
            device=self.device,
        ).to(self.device)

        # History buffer for raw observations (circular buffer)
        self.history_buffer = None
        self.current_step = 0

        # Storage for CENet training (will be initialized after env reset)
        self.cenet_storage = None

        # Attach runner to env for observation function access
        self.env.unwrapped.runner = self

        # Temporary storage for current CENet outputs (est_vel, context_vec)
        self.current_est_vel = None
        self.current_context_vec = None
        # Initialize history buffer for inference (can be overridden in learn)
        self._init_history_buffer(self.env.num_envs)

    def _init_history_buffer(self, num_envs: int):
        """Initialize history buffer with zeros."""
        self.history_buffer = torch.zeros(
            self.history_length,
            num_envs,
            self.raw_obs_dim,
            device=self.device,
            requires_grad=False,
        )
        self.current_step = 0

    def _init_cenet_storage(self):
        """Initialize CENet rollout storage."""
        num_envs = self.env.num_envs
        num_transitions_per_env = self.cfg["num_steps_per_env"]
        obs_history_shape = (self.history_length * self.raw_obs_dim,)
        true_vel_shape = (self.num_estvel,)
        true_onext_shape = (self.raw_obs_dim,)
        self.cenet.init_storage(
            num_envs,
            num_transitions_per_env,
            obs_history_shape,
            true_vel_shape,
            true_onext_shape,
        )

    def _get_true_velocity(self) -> torch.Tensor:
        """Retrieve true velocity of robot base (linear velocity in base frame)."""
        # Access robot entity from scene
        robot = self.env.unwrapped.scene["robot"]
        # root_link_lin_vel_b shape: (num_envs, 3)
        true_vel = robot.data.root_link_lin_vel_b
        return true_vel.to(self.device)

    def _get_actor_observation(self, obs: TensorDict) -> torch.Tensor:
        """Extract actor observation tensor from TensorDict."""
        # The TensorDict contains keys "actor" and "critic" (observation groups).
        return obs["actor"].to(self.device)

    def _get_raw_observation(self) -> torch.Tensor:
        """Get raw observation from environment (before any normalization)."""
        # The environment's get_observations returns a TensorDict with observation groups.
        obs = self.env.get_observations()
        actor_obs = self._get_actor_observation(obs)
        # raw_obs_dim corresponds to the first raw_obs_dim dimensions of actor observation
        raw_obs = actor_obs[:, :self.raw_obs_dim]
        return raw_obs

    def update_history(self, raw_obs: torch.Tensor):
        """Update history buffer with new raw observation."""
        if self.history_buffer is None:
            raise RuntimeError("History buffer not initialized")
        # raw_obs shape: (num_envs, raw_obs_dim)
        idx = self.current_step % self.history_length
        self.history_buffer[idx].copy_(raw_obs)
        self.current_step += 1

    def get_history_observations(self, env_ids: torch.Tensor, length: int) -> torch.Tensor:
        """Retrieve historical observations for given environment IDs.

        Args:
            env_ids: Tensor of environment indices.
            length: Number of past frames to retrieve (<= history_length).

        Returns:
            Tensor of shape (len(env_ids), length * raw_obs_dim)
        """
        if self.history_buffer is None:
            raise RuntimeError("History buffer not initialized")
        if length > self.history_length:
            raise ValueError(f"Requested length {length} exceeds history_length {self.history_length}")
        # Determine indices: most recent frames
        start_idx = (self.current_step - length) % self.history_length
        indices = [(start_idx + i) % self.history_length for i in range(length)]
        # shape (length, len(env_ids), raw_obs_dim)
        selected = self.history_buffer[indices][:, env_ids, :]
        # Transpose to (len(env_ids), length, raw_obs_dim) then flatten
        selected = selected.permute(1, 0, 2)
        batch_size = selected.shape[0]
        return selected.reshape(batch_size, -1)

    def inference_update_history(self, raw_obs: torch.Tensor):
        """Update history buffer with raw observation during inference.

        This should be called before each inference step to keep history buffer current.
        """
        if self.history_buffer is None:
            self._init_history_buffer(raw_obs.shape[0])
        self.update_history(raw_obs)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Run the learning loop with CENet integration."""
        # Randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Initialize history buffer and CENet storage
        num_envs = self.env.num_envs
        if self.history_buffer is None:
            self._init_history_buffer(num_envs)
        if self.cenet_storage is None:
            self._init_cenet_storage()

        # Start learning
        raw_obs = self._get_raw_observation()
        self.update_history(raw_obs)
        # Compute initial CENet features using current history (zeros for missing frames)
        obs_history = self.get_history_observations(
            torch.arange(num_envs, device=self.device), self.history_length
        )
        true_vel = self._get_true_velocity()
        _, est_vel, _, _, context_vec = self.cenet.before_action(obs_history, true_vel)
        self.current_est_vel = est_vel
        self.current_context_vec = context_vec
        # Get full observation (includes CENet features)
        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()  # switch to train mode (for dropout for example)
        self.cenet.train_mode()

        # Ensure all parameters are in-synced (for multi-GPU)
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Initialize the logging writer
        if self.logger.log_dir is not None:
            os.makedirs(self.logger.log_dir, exist_ok=True)
        self.logger.init_logging_writer()

        # Start training
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    # Retrieve history and true velocity for CENet
                    obs_history = self.get_history_observations(
                        torch.arange(num_envs, device=self.device), self.history_length
                    )
                    true_vel = self._get_true_velocity()

                    # CENet forward (before action)
                    _, est_vel, _, _, context_vec = self.cenet.before_action(
                        obs_history, true_vel
                    )
                    self.current_est_vel = est_vel
                    self.current_context_vec = context_vec

                    # Get augmented observation (includes CENet features)
                    obs = self.env.get_observations().to(self.device)
                    # Sample actions using augmented observations
                    actions = self.alg.act(obs)

                    # Step the environment
                    next_obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))

                    # Check for NaN values from the environment
                    if self.cfg.get("check_for_nan", True):
                        from rsl_rl.utils import check_nan
                        check_nan(next_obs, rewards, dones)

                    # Move to device
                    next_obs, rewards, dones = (
                        next_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )

                    # CENet after action (store next raw observation)
                    self.cenet.after_action(self._get_actor_observation(next_obs)[:, :self.raw_obs_dim])  # assume first raw_obs_dim are raw obs

                    # Update history buffer with new raw observation
                    self.update_history(self._get_actor_observation(next_obs)[:, :self.raw_obs_dim])

                    # Process the step for PPO
                    self.alg.process_env_step(next_obs, rewards, dones, extras)

                    # Book keeping
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

                    # Update obs for next iteration
                    obs = next_obs

                stop = time.time()
                collect_time = stop - start
                start = stop

                # Compute returns
                self.alg.compute_returns(obs)

            # CENet update (before PPO update)
            if it % self.cenet_update_freq == 0:
                mean_total_loss, mean_vel_loss, mean_recon_loss, mean_kl_loss = self.cenet.update()
                # Log CENet losses (optional)
                if self.logger.writer is not None:
                    self.logger.writer.add_scalar("CENet/total_loss", mean_total_loss, it)
                    self.logger.writer.add_scalar("CENet/vel_loss", mean_vel_loss, it)
                    self.logger.writer.add_scalar("CENet/recon_loss", mean_recon_loss, it)
                    self.logger.writer.add_scalar("CENet/kl_loss", mean_kl_loss, it)

            # PPO update
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # Log information
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
            )

            # Save model
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore

        # Save the final model after training and stop the logging writer
        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore
            self.logger.stop_logging_writer()

    def save(self, path: str, infos=None):
        # Ensure parent directory exists before any save operation
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # First call parent save to create the base checkpoint
        super().save(path, infos)
        # Load the saved checkpoint to add CENet weights
        checkpoint_dir = os.path.dirname(path)
        loaded_dict = torch.load(path, map_location=self.device, weights_only=False)
        # Add CENet state dict to the checkpoint
        loaded_dict["cenet_state_dict"] = self.cenet.state_dict()
        # Save back with CENet included
        torch.save(loaded_dict, path)
        # Export policy to ONNX (including CENet feature generation)
        filename = "policy.onnx"
        self.export_policy_to_onnx(checkpoint_dir, filename)
        run_name: str = (
            wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
        )  # type: ignore[assignment]
        onnx_path = os.path.join(checkpoint_dir, filename)
        metadata = get_base_metadata(self.env.unwrapped, run_name)
        attach_metadata_to_onnx(onnx_path, metadata)
        if self.logger.logger_type in ["wandb"]:
            wandb.save(os.path.join(checkpoint_dir, filename), base_path=checkpoint_dir)

    def load(self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None, **kwargs):
        # Call parent load and get the loaded dictionary
        loaded_dict = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location, **kwargs)
        # Load CENet state dict if present in the checkpoint
        if "cenet_state_dict" in loaded_dict:
            self.cenet.load_state_dict(loaded_dict["cenet_state_dict"])
            print(f"Loaded CENet from checkpoint {path}")
        else:
            print(f"CENet state dict not found in checkpoint {path}, skipping.")
        return loaded_dict

    def get_inference_policy(self, device: str | None = None):
        """Get inference policy with CENet feature integration.

        Overrides the base method to ensure history buffer is updated and CENet features
        are computed before each policy call.
        """
        # Get the original policy from parent
        policy = super().get_inference_policy(device)
        # Ensure history buffer is initialized
        if self.history_buffer is None:
            self._init_history_buffer(self.env.num_envs)
        # Return wrapped policy
        def wrapped_policy(obs):
            # obs is the observation returned by env.get_observations()
            # which already includes CENet features (if computed).
            # However, we need to update history buffer with raw observation
            # and compute CENet features for the next step.
            # For now, we just return the policy output.
            # TODO: Implement proper history update and CENet feature computation.
            return policy(obs)
        return wrapped_policy

    def export_policy_to_onnx(self, path: str, filename: str = "policy.onnx"):
        """Export policy to ONNX, incorporating CENet feature generation.

        This method should create a combined model that takes raw observations
        and history, runs CENet to produce features, then passes the augmented
        observations to the policy network.
        """
        # TODO: Implement ONNX export with CENet integration.
        # For now, fall back to parent's export (without CENet).
        super().export_policy_to_onnx(path, filename)