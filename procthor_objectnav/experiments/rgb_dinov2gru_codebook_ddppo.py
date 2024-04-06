try:
    from typing import final
except ImportError:
    from typing_extensions import final

import gym
import torch
import torch.nn as nn
from allenact.embodiedai.sensors.vision_sensors import DepthSensor, RGBSensor
from allenact_plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor

from procthor_objectnav.models.dino_gru import DinoTensorNavActorCriticCodebook
from procthor_objectnav.experiments.rgb_dinov2gru_ddppo import ProcTHORObjectNavRGBDINOv2PPOExperimentConfig


class ProcTHORObjectNavRGBDINOv2PPOCodebookExperimentConfig(ProcTHORObjectNavRGBDINOv2PPOExperimentConfig):
    """An Object Navigation experiment configuration with RGB input."""

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)

        self.MODEL = DinoTensorNavActorCriticCodebook

    @classmethod
    def tag(cls):
        return "ObjectNav-RGB-DINOv2GRU-DDPPO-Codebook"


    def create_model(self, **kwargs) -> nn.Module:
        has_rgb = any(isinstance(s, RGBSensor) for s in self.SENSORS)
        has_depth = any(isinstance(s, DepthSensor) for s in self.SENSORS)

        goal_sensor_uuid = next(
            (s.uuid for s in self.SENSORS if isinstance(s, GoalObjectTypeThorSensor)),
            None,
        )

        model = self.MODEL(
            action_space=gym.spaces.Discrete(len(self.cfg.mdp.actions)),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            goal_sensor_uuid=goal_sensor_uuid,
            rgb_dino_preprocessor_uuid="rgb_dinov2" if has_rgb else None,
            hidden_size=512,
            goal_dims=32,
            add_prev_actions=self.cfg.model.add_prev_actions_embedding,
            auxiliary_uuids=self.AUX_UUIDS,
            codebook_type=self.cfg.model.codebook.type,
            codebook_indexing=self.cfg.model.codebook.indexing,
            codebook_size=self.cfg.model.codebook.size,
            codebook_dim=self.cfg.model.codebook.code_dim,
            codebook_temperature=self.cfg.model.codebook.temperature,
            codebook_dropout_prob=self.cfg.model.codebook.dropout,
            codebook_embeds_type=self.cfg.model.codebook.embeds,
            codebook_topk=self.cfg.model.codebook.topk,
            visualize=self.cfg.visualize,
            task_id_uuid="task_id_sensor" if self.cfg.visualize else None,
        )

        if (
                self.cfg.pretrained_model.only_load_model_state_dict
                and self.cfg.checkpoint is not None
        ):
            if not torch.cuda.is_available():
                model.load_state_dict(
                    torch.load(self.cfg.checkpoint, map_location=torch.device("cpu"))[
                        "model_state_dict"
                    ],
                    strict=False,
                )  ## NOTE: strict=False does not work
            else:
                model.load_state_dict(
                    torch.load(self.cfg.checkpoint)["model_state_dict"], strict=False
                )  ## NOTE: strict=False does not work
        return model
