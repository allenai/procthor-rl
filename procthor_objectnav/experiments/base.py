import random
from abc import ABC
from math import ceil
from typing import Any, Dict, List, Optional, Sequence, Tuple

from torch.distributions.utils import lazy_property

from procthor_objectnav.tasks.object_nav import (
    FullProcTHORObjectNavTestTaskSampler,
    ProcTHORObjectNavTask,
    ProcTHORObjectNavTaskSampler,
)

try:
    from typing import Literal, final
except ImportError:
    from typing_extensions import Literal, final

import numpy as np
import prior
import torch
import torch.optim as optim
from ai2thor.platform import CloudRendering
from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.base_abstractions.experiment_config import ExperimentConfig, MachineParams
from allenact.base_abstractions.preprocessor import (
    Preprocessor,
    SensorPreprocessorGraph,
)
from allenact.base_abstractions.sensor import (
    ExpertActionSensor,
    Sensor,
    SensorSuite,
    Union,
)
from allenact.base_abstractions.task import TaskSampler
from allenact.embodiedai.sensors.vision_sensors import DepthSensor
from allenact.utils.experiment_utils import (
    Builder,
    PipelineStage,
    TrainingPipeline,
    TrainingSettings,
    evenly_distribute_count_into_bins,
)
from allenact.utils.system import get_logger
from omegaconf import OmegaConf
from procthor_objectnav.utils.types import RewardConfig, TaskSamplerArgs
from procthor_objectnav.callbacks.wandb_logging import SimpleWandbLogging


class ProcTHORObjectNavBaseConfig(ExperimentConfig, ABC):
    """The base config for all ObjectNav experiments."""

    DISTANCE_TYPE = "l2"  # "geo"  # Can be "geo" or "l2"

    RESAMPLE_SAME_SCENE_FREQ_IN_TRAIN = (
        -1
    )  # Should be > 0 if `ADVANCE_SCENE_ROLLOUT_PERIOD` is `None`
    RESAMPLE_SAME_SCENE_FREQ_IN_INFERENCE = 100

    OBJECT_NAV_TASK_TYPE = ProcTHORObjectNavTask

    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg

        self.SENSORS: Sequence[Sensor] = []

        # have evaluation tasks loaded
        self.EVALUATE = cfg.eval
        if self.EVALUATE:
            self.EVAL_TASKS = prior.load_dataset(
                "object-nav-eval",
                scene_datasets={task for task in cfg.evaluation.tasks},
                minival=cfg.evaluation.minival,
            )
        else:
            self.EVAL_TASKS = None

        if self.EVALUATE:
            if "robothor" in cfg.evaluation.tasks:
                if len(cfg.evaluation.tasks) == 1:
                    self.THOR_COMMIT_ID: Optional[str] = (
                        "bad5bc2b250615cb766ffb45d455c211329af17e"
                    )
                else:
                    raise NotImplementedError(
                        f"RoboTHOR evaluation should be evaluated by itself only with its own commit id"
                    )
            else:
                self.THOR_COMMIT_ID: Optional[str] = (
                    "ca10d107fb46cb051dba99af484181fda9947a28"
                )
        else:
            self.THOR_COMMIT_ID: Optional[str] = (
                "ca10d107fb46cb051dba99af484181fda9947a28"
            )

    def wandb_logging_callback(self) -> SimpleWandbLogging:
        return SimpleWandbLogging(
            project=self.cfg.wandb.project,
            entity=self.cfg.wandb.entity,
            name=self.cfg.wandb.name,
        )

    def get_devices(
        self, split: Literal["train", "valid", "test"]
    ) -> Tuple[torch.device]:
        if not torch.cuda.is_available():
            return (torch.device("cpu"),)

        if split == "train":
            gpus = self.cfg.machine.num_train_gpus
        elif split == "valid":
            gpus = self.cfg.machine.num_val_gpus
        elif split == "test":
            gpus = self.cfg.machine.num_test_gpus
        else:
            raise ValueError(f"Unknown split {split}")

        if gpus is None:
            gpus = torch.cuda.device_count()

        return tuple(torch.device(f"cuda:{i}") for i in range(gpus))

    @property
    def TRAIN_DEVICES(self) -> Tuple[torch.device]:
        return self.get_devices("train")

    @property
    def VAL_DEVICES(self) -> Tuple[torch.device]:
        return self.get_devices("valid")

    @property
    def TEST_DEVICES(self) -> Tuple[torch.device]:
        return self.get_devices("test")

    @lazy_property
    def HOUSE_DATASET(self):
        return prior.load_dataset("procthor-10k")

    def preprocessors(self) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        return tuple()

    @staticmethod
    def get_platform(
        gpu_index: int, platform: Literal["CloudRendering", "Linux64"]
    ) -> Dict[str, Any]:
        """Return the platform specific args to be passed into AI2-THOR.

        Parameters:
        - gpu_index: The index of the GPU to use. Must be in the range [0,
          torch.cuda.device_count() - 1].
        """
        if gpu_index < 0:
            return {}
        elif gpu_index >= torch.cuda.device_count():
            raise ValueError(
                f"gpu_index must be in the range [0, {torch.cuda.device_count()}]."
                f" You gave {gpu_index}."
            )

        if platform == "CloudRendering":
            return {"gpu_device": gpu_index, "platform": CloudRendering}
        elif platform == "Linux64":
            return {"x_display": f":0.{gpu_index}"}
        else:
            raise ValueError(f"Unknown platform: {platform}")

    def machine_params(self, mode: Literal["train", "valid", "test"], **kwargs):
        devices: Sequence[torch.device]
        nprocesses: int
        if mode == "train":
            devices = self.TRAIN_DEVICES * self.cfg.distributed.nodes
            nprocesses = (
                self.cfg.machine.num_train_processes * self.cfg.distributed.nodes
            )
        elif mode == "valid":
            devices = self.VAL_DEVICES
            nprocesses = self.cfg.machine.num_val_processes
        elif mode == "test":
            devices = self.TEST_DEVICES
            nprocesses = self.cfg.machine.num_test_processes
        else:
            raise NotImplementedError

        nprocesses = (
            evenly_distribute_count_into_bins(count=nprocesses, nbins=len(devices))
            if nprocesses > 0
            else [0] * len(devices)
        )

        sensors = [*self.SENSORS]
        if mode != "train":
            sensors = [s for s in sensors if not isinstance(s, ExpertActionSensor)]

        sensor_preprocessor_graph = (
            SensorPreprocessorGraph(
                source_observation_spaces=SensorSuite(sensors).observation_spaces,
                preprocessors=self.preprocessors(),
            )
            if mode == "train"
            or (
                (isinstance(nprocesses, int) and nprocesses > 0)
                or (isinstance(nprocesses, Sequence) and sum(nprocesses) > 0)
            )
            else None
        )

        params = MachineParams(
            nprocesses=nprocesses,
            devices=devices,
            sampler_devices=devices,
            sensor_preprocessor_graph=sensor_preprocessor_graph,
        )

        # NOTE: for distributed setup
        if mode == "train" and "machine_id" in kwargs:
            machine_id = kwargs["machine_id"]
            assert (
                0 <= machine_id < self.cfg.distributed.nodes
            ), f"machine_id {machine_id} out of range [0, {self.cfg.distributed.nodes} - 1]"
            local_worker_ids = list(
                range(
                    len(self.TRAIN_DEVICES) * machine_id,
                    len(self.TRAIN_DEVICES) * (machine_id + 1),
                )
            )
            params.set_local_worker_ids(local_worker_ids)

        return params

    def make_sampler_fn(
        self, task_sampler_args: TaskSamplerArgs, **kwargs
    ) -> TaskSampler:
        task_sampler_args.controller_args["server_timeout"] = 1000
        if self.EVALUATE:
            return FullProcTHORObjectNavTestTaskSampler(args=task_sampler_args)
        else:
            return ProcTHORObjectNavTaskSampler(
                args=task_sampler_args, object_nav_task_type=self.OBJECT_NAV_TASK_TYPE
            )

    @staticmethod
    def _partition_inds(n: int, num_parts: int):
        return np.round(
            np.linspace(start=0, stop=n, num=num_parts + 1, endpoint=True)
        ).astype(np.int32)

    def _get_sampler_args_for_scene_split(
        self,
        houses,
        mode: Literal["train", "eval"],
        resample_same_scene_freq: int,
        allow_oversample: bool,
        allow_flipping: bool,
        process_ind: int,
        total_processes: int,
        max_tasks: Optional[int],
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
        extra_controller_args: Optional[Dict[str, Any]] = None,
        visualize: bool = False,
    ) -> TaskSamplerArgs:
        # NOTE: oversample some scenes -> bias
        oversample_warning = (
            f"Warning: oversampling some of the houses ({houses}) to feed all processes ({total_processes})."
            " You can avoid this by setting a number of workers divisible by the number of scenes"
        )
        house_inds = list(range(len(houses)))
        if total_processes > len(houses):
            if not allow_oversample:
                raise RuntimeError(
                    f"Cannot have `total_processes > len(houses)`"
                    f" ({total_processes} > {len(houses)}) when `allow_oversample` is `False`."
                )

            if total_processes % len(houses) != 0:
                get_logger().warning(oversample_warning)
            house_inds = house_inds * ceil(total_processes / len(houses))
            house_inds = house_inds[
                : total_processes * (len(house_inds) // total_processes)
            ]
        elif len(houses) % total_processes != 0:
            if process_ind == 0:  # Only print warning once
                get_logger().warning(
                    f"Number of houses {len(houses)} is not cleanly divisible by the number"
                    f" of processes ({total_processes}). Because of this, not all processes will"
                    f" be fed the same number of houses."
                )

        inds = self._partition_inds(len(house_inds), total_processes)
        house_inds = house_inds[inds[process_ind] : inds[process_ind + 1]]

        controller_args = {
            # "branch": "nanna",
            "commit_id": self.THOR_COMMIT_ID,
            "width": self.cfg.agent.camera_width,
            "height": self.cfg.agent.camera_height,
            "rotateStepDegrees": self.cfg.agent.rotate_step_degrees,
            "visibilityDistance": self.cfg.agent.visibility_distance,
            "gridSize": self.cfg.agent.step_size,
            "agentMode": self.cfg.agent.agent_mode,
            "fieldOfView": self.cfg.agent.field_of_view,
            "snapToGrid": False,
            "renderDepthImage": any(isinstance(s, DepthSensor) for s in self.SENSORS),
            "renderInstanceSegmentation": False,
            **self.get_platform(
                gpu_index=devices[process_ind % len(devices)],
                platform=self.cfg.ai2thor.platform,
            ),
        }
        if extra_controller_args:
            controller_args.update(extra_controller_args)

        if max_tasks is not None and self.EVALUATE:
            max_tasks = min(len(house_inds), max_tasks)

        return TaskSamplerArgs(
            process_ind=process_ind,
            mode=mode,
            house_inds=house_inds,
            houses=houses,
            sensors=self.SENSORS,
            controller_args=controller_args,
            target_object_types=self.cfg.target_object_types,
            max_steps=self.cfg.mdp.max_steps,
            seed=seeds[process_ind] if seeds is not None else None,
            deterministic_cudnn=deterministic_cudnn,
            reward_config=RewardConfig(**self.cfg.mdp.reward[mode]),
            max_tasks=max_tasks if max_tasks is not None else len(house_inds),
            allow_flipping=allow_flipping,
            distance_type=self.DISTANCE_TYPE,
            resample_same_scene_freq=resample_same_scene_freq,
            # add to remove global config
            p_randomize_materials=self.cfg.procthor.p_randomize_materials,
            test_on_validation=self.cfg.evaluation.test_on_validation,
            actions=self.cfg.mdp.actions,
            max_agent_positions=self.cfg.training.object_selection.max_agent_positions,
            valid_agent_heights=self.cfg.agent.valid_agent_heights,
            max_vis_points=self.cfg.training.object_selection.max_vis_points,
            p_greedy_target_object=self.cfg.training.object_selection.p_greedy_target_object,
            ithor_p_shuffle_objects=self.cfg.ithor.p_shuffle_objects,
            visualize=visualize,
        )

    def train_task_sampler_args(self, **kwargs) -> Dict[str, Any]:
        train_houses = self.HOUSE_DATASET["train"]
        if self.cfg.procthor.num_train_houses:
            train_houses = train_houses.select(
                range(self.cfg.procthor.num_train_houses)
            )

        out = self._get_sampler_args_for_scene_split(
            houses=train_houses,
            mode="train",
            allow_oversample=True,
            max_tasks=float("inf"),
            allow_flipping=True,
            resample_same_scene_freq=self.RESAMPLE_SAME_SCENE_FREQ_IN_TRAIN,
            extra_controller_args=dict(
                branch="locobot-nanna-lookup", scene="Procedural"
            ),
            visualize=self.cfg.visualize,
            **kwargs,
        )
        return {"task_sampler_args": out}

    def valid_task_sampler_args(self, **kwargs) -> Dict[str, Any]:
        val_houses = self.HOUSE_DATASET["val"]
        out = self._get_sampler_args_for_scene_split(
            houses=(
                self.EVAL_TASKS["val"]
                if self.EVALUATE
                else val_houses.select(range(100))
            ),
            mode="eval",
            allow_oversample=False,
            max_tasks=(
                self.cfg.evaluation.max_val_tasks
                if self.EVALUATE
                else self.cfg.training.max_val_tasks
            ),
            allow_flipping=False,
            resample_same_scene_freq=self.RESAMPLE_SAME_SCENE_FREQ_IN_INFERENCE,
            extra_controller_args=None if self.EVALUATE else dict(scene="Procedural"),
            visualize=self.cfg.visualize,
            **kwargs,
        )
        return {"task_sampler_args": out}

    def test_task_sampler_args(self, **kwargs) -> Dict[str, Any]:
        if self.cfg.evaluation.test_on_validation:
            return self.valid_task_sampler_args(**kwargs)

        test_houses = self.HOUSE_DATASET["test"]
        out = self._get_sampler_args_for_scene_split(
            houses=(
                self.EVAL_TASKS["test"]
                if self.EVALUATE
                else test_houses.select(range(100))
            ),
            mode="eval",
            allow_oversample=False,
            max_tasks=(
                self.cfg.evaluation.max_test_tasks
                if self.EVALUATE
                else self.cfg.training.max_test_tasks
            ),
            allow_flipping=False,
            resample_same_scene_freq=self.RESAMPLE_SAME_SCENE_FREQ_IN_INFERENCE,
            extra_controller_args=None if self.EVALUATE else dict(scene="Procedural"),
            visualize=self.cfg.visualize,
            **kwargs,
        )
        return {"task_sampler_args": out}

    def training_pipeline(self, **kwargs):
        log_intervals = []
        batch_steps = []
        num_steps = []
        for n in range(self.cfg.training.num_stages):
            num_steps.append(self.cfg.training.base_num_steps * (2**n))
            if torch.cuda.is_available:
                log_intervals.append(
                    self.cfg.distributed.nodes
                    * self.cfg.machine.num_train_processes
                    * num_steps[-1]
                    * 5
                )
            else:
                log_intervals.append(1)
            if n < self.cfg.training.num_stages - 1:
                batch_steps.append(int(10e6))
            else:
                if len(batch_steps) > 0:
                    batch_steps.append(
                        int(self.cfg.training.ppo_steps - sum(batch_steps))
                    )
                else:
                    batch_steps.append(int(self.cfg.training.ppo_steps))

        assert num_steps[-1] == self.cfg.training.num_steps

        return TrainingPipeline(
            save_interval=self.cfg.training.save_interval,
            metric_accumulate_interval=self.cfg.training.log_interval,
            optimizer_builder=Builder(optim.Adam, dict(lr=self.cfg.training.lr)),
            num_mini_batch=self.cfg.training.num_mini_batch,
            update_repeats=self.cfg.training.update_repeats,
            max_grad_norm=self.cfg.training.max_grad_norm,
            num_steps=self.cfg.training.num_steps,
            named_losses={"ppo_loss": PPO(**PPOConfig)},
            gamma=self.cfg.training.gamma,
            use_gae=self.cfg.training.use_gae,
            gae_lambda=self.cfg.training.gae_lambda,
            advance_scene_rollout_period=self.cfg.training.advance_scene_rollout_period,
            pipeline_stages=[
                PipelineStage(
                    loss_names=["ppo_loss"],
                    max_stage_steps=batch_steps[n],
                    training_settings=TrainingSettings(
                        num_steps=num_steps[n],
                        metric_accumulate_interval=log_intervals[n],
                    ),
                )
                for n in range(self.cfg.training.num_stages)
            ],
        )
