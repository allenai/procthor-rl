from typing import Sequence, Union

import gym
import torch.nn as nn

from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.embodiedai.sensors.vision_sensors import DepthSensor, RGBSensor
from allenact.utils.experiment_utils import Builder
from allenact_plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor
from allenact_plugins.navigation_plugin.objectnav.models import ObjectNavActorCritic
from allenact_plugins.robothor_plugin.robothor_tasks import ObjectNavTask
from procthor_objectnav import cfg
from .base import ProcTHORObjectNavBaseConfig
from ..sensors.vision import RGBSensorThorController


class ProcTHORObjectNavRGBClipResNet50PPOExperimentConfig(ProcTHORObjectNavBaseConfig):
    """An Object Navigation experiment configuration with RGB input."""

    SENSORS = [
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

    BACKBONE = (
        # "gnresnet18"
        "simple_cnn"
    )

    @classmethod
    def preprocessors(cls) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        return []

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        rgb_uuid = next((s.uuid for s in cls.SENSORS if isinstance(s, RGBSensor)), None)
        depth_uuid = next(
            (s.uuid for s in cls.SENSORS if isinstance(s, DepthSensor)), None
        )
        goal_sensor_uuid = next(
            (s.uuid for s in cls.SENSORS if isinstance(s, GoalObjectTypeThorSensor)),
            None,
        )

        return ObjectNavActorCritic(
            action_space=gym.spaces.Discrete(len(ObjectNavTask.class_action_names())),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            rgb_uuid=rgb_uuid,
            depth_uuid=depth_uuid,
            goal_sensor_uuid=goal_sensor_uuid,
            hidden_size=512,
            backbone=cls.BACKBONE,
            resnet_baseplanes=32,
            object_type_embedding_dim=32,
            num_rnn_layers=1,
            rnn_type="GRU",
            add_prev_actions=cfg.model.add_prev_actions_embedding,
        )

    @classmethod
    def tag(cls):
        return "ObjectNav-RGB-SimpleGRU-DDPPO"
