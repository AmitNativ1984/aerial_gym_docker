from aerial_gym.config.sensor_config.camera_config.base_depth_camera_config import (
    BaseDepthCameraConfig,
)


class D435DepthCameraConfig(BaseDepthCameraConfig):
    """Intel RealSense D435 depth camera configuration.

    Specs: 87 deg HFOV, ~58 deg VFOV (derived from aspect ratio), 1280x720 resolution.
    """

    num_sensors = 1
    sensor_type = "camera"

    # Intel RealSense D435 resolution and FOV
    height = 720
    width = 1280
    horizontal_fov_deg = 87.0
    # VFOV = 2 * atan(tan(87/2 * pi/180) * (720/1280)) ~ 58.3 deg (matches D435 spec)

    max_range = 10.0  # meters
    min_range = 0.105  # D435 min depth

    calculate_depth = True
    return_pointcloud = False
    pointcloud_in_world_frame = False
    segmentation_camera = False  # not needed for VAE dataset

    euler_frame_rot_deg = [-90.0, 0, -90.0]

    normalize_range = True
    normalize_range = (
        False
        if (return_pointcloud == True and pointcloud_in_world_frame == True)
        else normalize_range
    )

    far_out_of_range_value = max_range if normalize_range == True else -1.0
    near_out_of_range_value = -max_range if normalize_range == True else -1.0

    # Disable sensor placement randomization -- we randomize the whole robot pose instead
    randomize_placement = False
    min_translation = [0.10, 0.0, 0.03]
    max_translation = [0.10, 0.0, 0.03]
    min_euler_rotation_deg = [0.0, 0.0, 0.0]
    max_euler_rotation_deg = [0.0, 0.0, 0.0]

    nominal_position = [0.10, 0.0, 0.03]
    nominal_orientation_euler_deg = [0.0, 0.0, 0.0]

    use_collision_geometry = False

    class sensor_noise:
        enable_sensor_noise = False
        pixel_dropout_prob = 0.0
        pixel_std_dev_multiplier = 0.0
