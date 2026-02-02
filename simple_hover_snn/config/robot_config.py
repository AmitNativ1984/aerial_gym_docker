"""
Robot configuration used for simple hover SNN task.

NOTE: This file is kept for reference but is NOT used in training.
The task now uses "base_quadrotor" directly (matching position_setpoint_task).

See simple_hover_snn/config/task_config.py: robot_name = "base_quadrotor"
"""

import numpy as np
from aerial_gym.config.robot_config.base_quad_config import BaseQuadCfg

# This class is no longer used - task_config specifies robot_name = "base_quadrotor"
class CustomQuadWithImuSNNCfg(BaseQuadCfg):
    """
    DEPRECATED: Task now uses base_quadrotor directly.

    This config is kept for reference only.
    """
    pass
