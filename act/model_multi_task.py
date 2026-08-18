import cv2
import numpy as np
import torch
from .detr.act_policy import ACT
from argparse import Namespace

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import pack_robot_state, unpack_robot_state, get_robot_action_dim_info
import os


class Model(ModelTemplate):

    def __init__(self, model_cfg):
        self.camera_names = model_cfg.get('camera_names', [])
        model_cfg['camera_names'] = self.camera_names

        self._model_cfg = dict(model_cfg)
        self._current_task = None

        self.model = self.get_model(model_cfg=model_cfg)
        self.robot_action_dim_info = get_robot_action_dim_info(model_cfg['env_cfg_type'])
        self.action_type = model_cfg['action_type']

    def get_model(self, model_cfg):
        if not model_cfg.get('ckpt_dir'):
            if not model_cfg.get('ckpt_name'):
                raise ValueError("ACT requires ckpt_name or ckpt_dir during evaluation.")
            model_cfg['ckpt_dir'] = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'checkpoints', str(model_cfg['ckpt_name']))
        return ACT(model_cfg, Namespace(**model_cfg))

    def _resolve_task_ckpt_dir(self, task_name):
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
        candidates = [os.path.join(base, "act-RoboDojo-" + task_name)]
        if not task_name.startswith("act-"):
            candidates.append(os.path.join(base, task_name))
        for task_dir in candidates:
            if not os.path.isdir(task_dir):
                continue
            for entry in sorted(os.listdir(task_dir)):
                subdir = os.path.join(task_dir, entry)
                if (os.path.isfile(os.path.join(subdir, 'policy_last.ckpt'))
                        and os.path.isfile(os.path.join(subdir, 'dataset_stats.pkl'))):
                    return subdir
            if os.path.isfile(os.path.join(task_dir, 'policy_last.ckpt')):
                return task_dir
        raise FileNotFoundError("ACT checkpoint dir not found for task '{0}' under {1}".format(task_name, base))

    def prepare_case(self, case_meta=None):
        case_meta = case_meta or {}
        task_name = (
            case_meta.get("task_name")
            or case_meta.get("bench_name")
            or case_meta.get("action_case_id")
            or ""
        )
        if isinstance(task_name, str) and task_name.endswith("_case"):
            task_name = task_name[:-len("_case")]
        task_name = str(task_name or "").strip()
        if not task_name:
            return
        if task_name == self._current_task:
            self.reset()
            return
        ckpt_dir = self._resolve_task_ckpt_dir(task_name)
        model_cfg = dict(self._model_cfg)
        model_cfg["ckpt_dir"] = ckpt_dir
        self.model = self.get_model(model_cfg=model_cfg)
        self._current_task = task_name
        print("[ACT] prepare_case -> task={0} ckpt={1}".format(task_name, ckpt_dir), flush=True)
        self.reset()

    def update_obs(self, obs):
        encoded_obs = self.encode_obs(obs, self.action_type, self.robot_action_dim_info)
        self.model.update_obs(encoded_obs)

    def get_action(self):
        actions = self.model.get_action()
        action_list = unpack_robot_state(actions, self.action_type, self.robot_action_dim_info, source_type="obs")
        return action_list

    def reset(self):
        if self.model.temporal_agg:
            self.model.all_time_actions = torch.zeros([
                self.model.max_timesteps,
                self.model.max_timesteps + self.model.num_queries,
                self.model.state_dim,
            ]).to(self.model.device)
            self.model.t = 0
        else:
            self.model.t = 0

    def encode_obs(self, observation, action_type, robot_action_dim_info):
        res_dict = dict()

        for camera_name in self.camera_names:
            if camera_name not in observation["vision"]:
                raise ValueError("Expected camera '{0}' not found in observation['vision']".format(camera_name))
            color = cv2.resize(observation["vision"][camera_name]["color"], (640, 480), interpolation=cv2.INTER_LINEAR)
            color = np.moveaxis(color, -1, 0) / 255.0
            res_dict[camera_name] = color

        res_dict["qpos"] = pack_robot_state(observation, action_type, robot_action_dim_info, source_type="obs")

        return res_dict
