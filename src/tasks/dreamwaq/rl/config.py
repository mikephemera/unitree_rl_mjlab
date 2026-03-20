"""DreamWaq-specific RL configuration."""

from dataclasses import dataclass, field
from typing import Literal, Tuple

from mjlab.rl import RslRlOnPolicyRunnerCfg


@dataclass
class DreamWaqOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Runner configuration for DreamWaq tasks.

    Extends the standard RSL-RL configuration with CENet hyperparameters
    and history buffer settings.
    """

    # CENet hyperparameters
    cenet_beta: float = 1.0
    """KL divergence weight for VAE beta-VAE loss (default 1.0)."""
    cenet_beta_limit: float = 4.0
    """Upper limit for beta (default 4.0)."""
    cenet_learning_rate: float = 0.01
    """Learning rate for CENet optimizer (default 0.01)."""
    cenet_optimizer: Literal["adam", "adamw"] = "adam"
    """Optimizer for CENet (default 'adam')."""

    # History buffer
    history_length: int = 5
    """Number of past observation frames to store (default 5)."""
    num_context: int = 16
    """Dimensionality of context vector (default 16)."""
    num_estvel: int = 3
    """Dimensionality of estimated velocity (default 3)."""

    # Observation dimensions (derived from environment)
    raw_obs_dim: int = 45
    """Raw observation dimension per timestep (default 45)."""

    # Optional: curriculum for beta annealing
    cenet_beta_anneal_steps: int = 0
    """Number of steps over which to linearly anneal beta to beta_limit (default 0 = no annealing)."""

    # Optional: CENet update frequency (steps per PPO update)
    cenet_update_freq: int = 1
    """Number of CENet updates per PPO update (default 1)."""

    # Architecture (can be overridden)
    cenet_hidden_dims: Tuple[int, ...] = field(default_factory=lambda: (256, 128))
    """Hidden dimensions for CENet encoder/decoder (default (256, 128))."""