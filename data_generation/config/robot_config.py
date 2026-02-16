import numpy as np

from aerial_gym.config.robot_config.base_quad_config import BaseQuadCfg
from data_generation.config.camera_config import D435DepthCameraConfig


class DataGenQuadCfg(BaseQuadCfg):
    """Quadrotor used as a camera carrier for dataset generation.

    Wide position and orientation randomization to capture diverse viewpoints.
    Gravity disabled so the robot floats in place between resets.
    """

    class init_config:
        # [ratio_x, ratio_y, ratio_z, roll, pitch, yaw, 1.0, vx, vy, vz, wx, wy, wz]
        min_init_state = [
            0.1, 0.1, 0.1,                      # position ratios (10-90% of env bounds)
            -np.pi / 3, -np.pi / 3, -np.pi,     # roll ±60 deg, pitch ±60 deg, yaw full
            1.0,
            0.0, 0.0, 0.0,                      # zero linear velocity
            0.0, 0.0, 0.0,                      # zero angular velocity
        ]
        max_init_state = [
            0.9, 0.9, 0.9,
            np.pi / 3, np.pi / 3, np.pi,
            1.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ]

    class sensor_config:
        enable_camera = True
        camera_config = D435DepthCameraConfig

        enable_lidar = False
        lidar_config = None

        enable_imu = False
        imu_config = None

    class disturbance:
        enable_disturbance = False
        prob_apply_disturbance = 0.0
        max_force_and_torque_disturbance = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    class damping(BaseQuadCfg.damping):
        pass

    class robot_asset(BaseQuadCfg.robot_asset):
        disable_gravity = True
        fix_base_link = False
        collision_mask = 0

        min_state_ratio = [
            0.1, 0.1, 0.1,
            -np.pi / 3, -np.pi / 3, -np.pi,
            1.0,
            0, 0, 0,
            0, 0, 0,
        ]
        max_state_ratio = [
            0.9, 0.9, 0.9,
            np.pi / 3, np.pi / 3, np.pi,
            1.0,
            0, 0, 0,
            0, 0, 0,
        ]

    class control_allocator_config(BaseQuadCfg.control_allocator_config):
        pass
