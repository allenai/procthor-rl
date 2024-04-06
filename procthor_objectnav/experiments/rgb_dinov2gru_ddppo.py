try:
    from typing import final
except ImportError:
    from typing_extensions import final

from typing import Sequence, Union

from procthor_objectnav.preprocessors.dino_preprocessors import DinoViTPreprocessor

import gym
import numpy as np
import torch
import torch.nn as nn
from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.embodiedai.sensors.vision_sensors import DepthSensor, RGBSensor
from allenact_plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor
from procthor_objectnav.sensors.log_sensor import TaskIdSensor
from procthor_objectnav.models.dino_gru import DinoTensorNavActorCritic

from ..sensors.vision import RGBSensorThorController
from .base import ProcTHORObjectNavBaseConfig
from allenact.utils.experiment_utils import Builder


class ProcTHORObjectNavRGBDINOv2PPOExperimentConfig(ProcTHORObjectNavBaseConfig):
    """An Object Navigation experiment configuration with RGB input."""

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.SENSORS = [
            RGBSensorThorController(
                height=cfg.model.image_size,
                width=cfg.model.image_size,
                use_resnet_normalization=True,
                uuid="rgb_lowres",
                mean=cfg.model.rgb_means,
                stdev=cfg.model.rgb_stds,
            ),
            GoalObjectTypeThorSensor(
                object_types=cfg.target_object_types,
            ),
        ]

        if self.cfg.visualize:
            self.SENSORS.append(
                TaskIdSensor(uuid="task_id_sensor")
            )

        self.AUX_UUIDS = []

        self.MODEL = DinoTensorNavActorCritic
        self.DINO_MODEL_TYPE = "dinov2_vits14"

    @classmethod
    def tag(cls):
        return "ObjectNav-RGB-DINOv2GRU-DDPPO"

    def preprocessors(self) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        preprocessors = []

        rgb_sensor = next((s for s in self.SENSORS if isinstance(s, RGBSensor)), None)
        assert (
            np.linalg.norm(
                np.array(rgb_sensor._norm_means) - np.array(self.cfg.model.rgb_means)
            )
            < 1e-5
        )
        assert (
            np.linalg.norm(
                np.array(rgb_sensor._norm_sds) - np.array(self.cfg.model.rgb_stds)
            )
            < 1e-5
        )

        if rgb_sensor is not None:
            preprocessors.append(
                DinoViTPreprocessor(
                    rgb_input_uuid=rgb_sensor.uuid,
                    dino_model_type=self.DINO_MODEL_TYPE,  # TODO: Standardize this
                    output_uuid="rgb_dinov2",
                    class_emb_only=True,
                    input_img_height_width=(
                        self.cfg.model.image_size,
                        self.cfg.model.image_size,
                    ),
                    chunk_size=64,
                    flatten=False,
                )
            )

        depth_sensor = next(
            (s for s in self.SENSORS if isinstance(s, DepthSensor)), None
        )
        assert depth_sensor is None

        return preprocessors

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
