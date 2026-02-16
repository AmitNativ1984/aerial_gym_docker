import trimesh
from urdfpy import Cylinder, Box, Sphere

# Monkey-patch urdfpy primitive geometry classes.
# Bug: _meshes is initialized to None, so len(self._meshes) raises TypeError.
# Cylinder also has a typo: returns self._mesh instead of self._meshes.


@property
def _cylinder_meshes(self):
    if self._meshes is None or len(self._meshes) == 0:
        self._meshes = [trimesh.creation.cylinder(
            radius=self.radius, height=self.length
        )]
    return self._meshes


@property
def _box_meshes(self):
    if self._meshes is None or len(self._meshes) == 0:
        self._meshes = [trimesh.creation.box(extents=self.size)]
    return self._meshes


@property
def _sphere_meshes(self):
    if self._meshes is None or len(self._meshes) == 0:
        self._meshes = [trimesh.creation.icosphere(radius=self.radius)]
    return self._meshes


Cylinder.meshes = _cylinder_meshes
Box.meshes = _box_meshes
Sphere.meshes = _sphere_meshes

# Register custom configs with aerial_gym registries
from aerial_gym.registry.env_registry import env_config_registry
from aerial_gym.registry.robot_registry import robot_registry
from aerial_gym.robots.base_multirotor import BaseMultirotor

from data_generation.config.env_config import DataGenEnvCfg
from data_generation.config.robot_config import DataGenQuadCfg

env_config_registry.register("data_gen_env", DataGenEnvCfg)
robot_registry.register("data_gen_quad", BaseMultirotor, DataGenQuadCfg)
