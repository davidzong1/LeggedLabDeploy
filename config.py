# Copyright (c) 2022-2025, The unitree_rl_gym Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from unitree_rl_gym Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).

import numpy as np
import yaml


class Config:
    def __init__(self, common_path, depoly_config_path) -> None:
        # Load common configuration
        with open(common_path) as f:
            config: dict = yaml.load(f, Loader=yaml.FullLoader)
            self.robot_name = config["robot_name"]
            self.msg_type = config["msg_type"]
            self.imu_type = config["imu_type"]
            self.joint2motor_idx = config["joint2motor_idx"]
            self.polic_idx = config["policy_action_ids"]
            self.kps = config["KP"]
            self.kds = config["KD"]
            self.weak_motor = []
            if "weak_motor" in config:
                self.weak_motor = config["weak_motor"]
            if "use_sportstate" in config:
                self.use_sportstate = config["use_sportstate"]
                self.urdf_path = config["urdf_path"]
            else:
                self.use_sportstate = False
                self.urdf_path = None

            self.lowcmd_topic = config["lowcmd_topic"]
            self.lowstate_topic = config["lowstate_topic"]
            self.sportstate_topic = config["sportstate_topic"]
            self.policy_path = config["policy_path"]
            self.total_action = config["total_action"]
            self.command_range = config["command_range"]
            joint_threshold_cache: dict = config["joint_threshold"]
            self.joint_threshold = []
            for key, value in joint_threshold_cache.items():
                self.joint_threshold.append(np.array(value, dtype=np.float32))

        # load policy depoly configuration
        with open(depoly_config_path) as f:
            config: dict = yaml.load(f, Loader=yaml.FullLoader)
            self.control_dt = config["step_dt"]
            self.default_joint_pos = np.array(
                config["default_joint_pos"], dtype=np.float32
            )
            if "torso_idx" in config:
                self.torso_idx = config["torso_idx"]

            if self.robot_name == "g1":
                self.policy_obs = config["observation"]
                self.action_scale = config["actions"]["scale"]

            self.base_policy_history_len = config["observations"]["projected_gravity"][
                "history_length"
            ]
            self.num_obs = 0
            for k, v in config["observations"].items():
                self.num_obs += len(v["scale"])
            self.num_actions = len(config["actions"]["scale"])
