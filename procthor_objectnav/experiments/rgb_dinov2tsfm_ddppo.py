import torch.cuda
import os

from ..models.dino_transformer import DinoTransformerNavActorCritic
from ..sensors.sequence_sensor import TimeStepSensor, TrajectorySensor

try:
    from typing import final
except ImportError:
    from typing_extensions import final

from typing import Sequence, Union

from procthor_objectnav.preprocessors.dino_preprocessors import DinoViTPreprocessor

import gym
import numpy as np
import torch.nn as nn
from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.embodiedai.sensors.vision_sensors import DepthSensor, RGBSensor
from allenact_plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor
from procthor_objectnav.sensors.log_sensor import TaskIdSensor

from ..sensors.vision import RGBSensorThorController
from .base import ProcTHORObjectNavBaseConfig
from allenact.utils.experiment_utils import Builder


class ProcTHORObjectNavRGBDINOv2TransformerDecoderPPOExperimentConfig(
    ProcTHORObjectNavBaseConfig
):
    """An Object Navigation experiment configuration with RGB input."""

    TRAJ_MAX_INDEX = 2048  # just a large number
    USE_ATTN_MASK: bool

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)

        self.USE_ATTN_MASK = cfg.training.use_attn_mask

        self.SENSORS = [
            RGBSensorThorController(
                height=cfg.model.image_size,
                width=cfg.model.image_size,
                use_resnet_normalization=True,
                uuid="rgb_lowres",
                mean=cfg.model.rgb_means,
                stdev=cfg.model.rgb_stds,
            ),
            GoalObjectTypeThorSensor(object_types=cfg.target_object_types,),
            TimeStepSensor(uuid="time_step", max_time_for_random_shift=0),
            TrajectorySensor(uuid="traj_index", max_idx=self.TRAJ_MAX_INDEX),
        ]

        if self.cfg.visualize:
            self.SENSORS.append(TaskIdSensor(uuid="task_id_sensor"))

        self.AUX_UUIDS = []

        self.MODEL = DinoTransformerNavActorCritic
        self.DINO_MODEL_TYPE = cfg.model.dino.model_type

    @classmethod
    def tag(cls):
        if os.getenv("TAG") is not None:
            return os.getenv("TAG")
        else:
            return "ObjectNav-RGB-DINOv2-TSFM-DDPPO"

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
        has_rgb = True
        goal_sensor_uuid = next(
            (s.uuid for s in self.SENSORS if isinstance(s, GoalObjectTypeThorSensor)),
            None,
        )

        max_gpus = torch.cuda.device_count()
        max_nproc = max(
            [
                n_proc // n_gpus
                for n_proc, n_gpus in zip(
                    [
                        self.cfg.machine.num_train_processes,
                        self.cfg.machine.num_val_processes,
                        self.cfg.machine.num_test_processes,
                    ],
                    [
                        self.cfg.machine.num_train_gpus or max_gpus or 1,
                        self.cfg.machine.num_val_gpus or max_gpus or 1,
                        self.cfg.machine.num_test_gpus or max_gpus or 1,
                    ],
                )
            ]
        )
        model: DinoTransformerNavActorCritic = self.MODEL(
            action_space=gym.spaces.Discrete(len(self.cfg.mdp.actions)),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            goal_sensor_uuid=goal_sensor_uuid,
            rgb_dino_preprocessor_uuid="rgb_dinov2" if has_rgb else None,
            depth_preprocessor_uuid=None,
            relevant_object_box_uuid=None,
            traj_idx_uuid="traj_index" if self.USE_ATTN_MASK else None,
            time_step_uuid="time_step",
            add_prev_action_null_token=True,
            add_prev_actions=self.cfg.model.add_prev_actions_embedding,
            auxiliary_uuids=self.AUX_UUIDS,
            initial_tgt_cache_shape=(self.cfg.mdp.max_steps, max_nproc),
            max_steps=self.cfg.mdp.max_steps,
            use_transformer_encoder=self.cfg.training.use_transformer_encoder,
            visualize=self.cfg.visualize,
            task_id_uuid="task_id_sensor",
            **self.cfg.model.nn_kwargs
        )

        return model
