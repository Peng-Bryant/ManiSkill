from typing import Any, Dict, Union
import numpy as np
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import Fetch, Panda, Xmate3Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig

@register_env("PushCubeTraj", max_episode_steps=50)
class PushCubeTrajEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda", "xmate3_robotiq", "fetch"]
    cube_half_size = 0.02
    goal_radius = 0.02
    reach_threshold = 0.8  # Success threshold for reach rate

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.trajectory = self.generate_trajectory()
        self.current_waypoint_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reached_waypoints = torch.zeros((self.num_envs, len(self.trajectory)), dtype=torch.bool, device=self.device)

    def generate_trajectory_sin(self):
        # Generate waypoints for a sine wave trajectory as an example
        num_waypoints = 40
        x = torch.linspace(-0.1, 0.5, num_waypoints)
        y = 0.1 * torch.sin(2 * np.pi * x)
        waypoints = torch.stack([x, y], dim=1)
        return waypoints

    def generate_trajectory_line(self):
        # Generate waypoints for a straight line trajectory as an example
        num_waypoints = 40
        x = torch.linspace(-0.1, 0.3, num_waypoints)
        y = torch.zeros(num_waypoints)
        waypoints = torch.stack([x, y], dim=1)
        return waypoints

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_cfg=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.obj = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=np.array([12, 42, 160, 255]) / 255,
            name="cube",
            body_type="dynamic",
        )

        # Visualize waypoints as red spheres
        self.waypoints = []
        for wp in self.trajectory:
            waypoint_sphere = actors.build_sphere(
                self.scene,
                radius=0.01,
                color=np.array([255, 0, 0, 255]) / 255,
                name="waypoint",
                body_type="kinematic",
            )
            waypoint_pose = Pose.create_from_pq(p=wp, q=[1, 0, 0, 0])
            waypoint_sphere.set_pose(waypoint_pose)
            self.waypoints.append(waypoint_sphere)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            start_pose = self.trajectory[0]
            xyz = torch.zeros((b, 3), device=self.device)
            xyz[..., :2] = start_pose
            xyz[..., 2] = self.cube_half_size
            q = torch.tensor([1, 0, 0, 0], device=self.device)
            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.obj.set_pose(obj_pose)

            self.current_waypoint_index = torch.zeros(b, dtype=torch.long, device=self.device)
            self.reached_waypoints = torch.zeros((b, len(self.trajectory)), dtype=torch.bool, device=self.device)

    def evaluate(self):
        cube_pos = self.obj.pose.p[..., :2]  # (num_envs, 2)
        waypoint_pos = self.trajectory.to(self.device)  # (num_waypoints, 2)
        distances = torch.cdist(cube_pos, waypoint_pos)  # (num_envs, num_waypoints)
        self.reached_waypoints = distances < self.goal_radius

        reach_rates = self.reached_waypoints.float().mean(dim=1)  # (num_envs,)
        success = reach_rates >= self.reach_threshold

        return {"success": success.cpu().numpy(), "reach_rate": reach_rates.cpu().numpy()}

    def _get_obs_extra(self, info: Dict):
        next_waypoints = self.trajectory[self.current_waypoint_index].cpu().numpy()
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            next_waypoint=next_waypoints,
        )
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                goal_pos=next_waypoints,
                obj_pose=self.obj.pose.raw_pose,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        waypoint = self.trajectory[self.current_waypoint_index]
        cube_pos = self.obj.pose.p[..., :2]
        distance = torch.norm(cube_pos - waypoint, dim=1)
        reward = -distance

        mask = distance < self.goal_radius
        self.current_waypoint_index = torch.min(self.current_waypoint_index + mask.long(), torch.tensor(len(self.trajectory) - 1, device=self.device))
        reward += mask.float() * 10  # Reward for reaching the waypoint

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        max_reward = 10.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
