"""DreamWaq task for quadruped locomotion on rough terrain.

This package implements the DreamWaq algorithm with CENet (Context-Aided Estimator Network)
for Unitree robots in the mjlab framework.
"""

from . import mdp
from . import rl
from . import config
from .dreamwaq_env_cfg import make_dreamwaq_env_cfg

__all__ = ["make_dreamwaq_env_cfg", "mdp", "rl", "config"]