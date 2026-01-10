"""
Robot configuration used for simple hover task.
Uses only an IMU onboard the quadrotor.

Controls the following robot parameters:
- Spawn position adjusted to be above ground plane
- Initial linear and angular velocities
"""

import numpy as np
from aerial_gym.config.sensor_config.imu_config.bosch_bmi088_config import BoschBMI088Config
from aerial_gym.config.robot_config.base_quad_config import BaseQuadCfg

class CustomQuadWithImuCfg(BaseQuadCfg):
    """
    Quadrotor with IMU for hover task.
    Spawns near center of environment:
        - Above ground plane
        - None-zero initial velocities
        - None-zero initial angular velocities
        - Random yaw orientation
        
    """
    class init_config(BaseQuadCfg.init_config):
        # initial state tensor:
        # [ratio_x, ratio_y, ratio_z, 
        # roll_radians, pitch_radians, yaw_radians,
        # lin_vel_x, lin_vel_y, lin_vel_z,
        # ang_vel_x, ang_vel_y, ang_vel_z]

        min_state_ratio = [
            0.45, 0.45, 0.4,            # position x, y, z: near center, above ground
            -np.pi/6, -np.pi/6, 0.0,    # rotation: roll, pitch, yaw
            1.0,                        # scale
            -1.0, -1.0, -0.5,           # linear velocity vx, vy, vz
            -0.5, -0.5, -0.5            # angular velocity wx, wy, wz
        ]


        max_state_ratio = [
            0.55, 0.55, 0.6,            # position x, y, z: near center, above ground
            np.pi/6, np.pi/6, 2.0*np.pi, # rotation: roll, pitch, yaw
            1.0,                        # scale
            1.0, 1.0, 0.5,              # linear velocity vx, vy, vz
            0.5, 0.5, 0.5               # angular velocity wx, wy, wz
        ]

    class sensor_config(BaseQuadCfg.sensor_config):
        enable_imu = True
        imu_config = BoschBMI088Config