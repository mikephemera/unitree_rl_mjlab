"""RL configuration for Unitree Go2 DreamWaq task."""

from src.tasks.dreamwaq.rl.config import DreamWaqOnPolicyRunnerCfg
from mjlab.rl import (
    RslRlModelCfg,
    RslRlPpoAlgorithmCfg,
)


def unitree_go2_dreamwaq_runner_cfg() -> DreamWaqOnPolicyRunnerCfg:
    """Create RL runner configuration for Unitree Go2 DreamWaq task."""
    return DreamWaqOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="go2_dreamwaq",
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=10001,
        # DreamWaq-specific hyperparameters
        cenet_beta=1.0,
        cenet_beta_limit=4.0,
        cenet_learning_rate=0.01,
        cenet_optimizer="adam",
        history_length=5,
        num_context=16,
        num_estvel=3,
        raw_obs_dim=45,
        cenet_beta_anneal_steps=0,
        cenet_update_freq=1,
        cenet_hidden_dims=(256, 128),
    )