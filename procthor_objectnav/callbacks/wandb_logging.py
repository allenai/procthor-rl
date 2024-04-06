from typing import Any, Dict, List, Sequence, Optional, Set
import numpy as np
import gym
from PIL import Image
from allenact.base_abstractions.sensor import Sensor
from procthor_objectnav.callbacks.local_logging import LocalLoggingSensor

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import wandb
from allenact.base_abstractions.callbacks import Callback


class SimpleWandbLogging(Callback):
    def __init__(
        self,
        project: str,
        entity: str,
        name: str,
    ):
        self.project = project
        self.entity = entity
        self.name = name

        self._defined_metrics: Set[str] = set()

    def setup(self, name: str, **kwargs) -> None:
        wandb.init(
            project=self.project,
            entity=self.entity,
            name=self.name,
            config=kwargs,
        )

    def _define_missing_metrics(
        self,
        metric_means: Dict[str, float],
        scalar_name_to_total_experiences_key: Dict[str, str],
    ):
        for k, v in metric_means.items():
            if k not in self._defined_metrics:
                wandb.define_metric(
                    k,
                    step_metric=scalar_name_to_total_experiences_key.get(
                        k, "training_step"
                    ),
                )

                self._defined_metrics.add(k)

    def on_train_log(
        self,
        metrics: List[Dict[str, Any]],
        metric_means: Dict[str, float],
        step: int,
        tasks_data: List[Any],
        scalar_name_to_total_experiences_key: Dict[str, str],
        **kwargs,
    ) -> None:
        """Log the train metrics to wandb."""

        self._define_missing_metrics(
            metric_means=metric_means,
            scalar_name_to_total_experiences_key=scalar_name_to_total_experiences_key,
        )

        wandb.log(
            {
                **metric_means,
                "training_step": step,
            }
        )

    def combine_rgb_across_episode(self, observation_list):
        all_rgb = []
        for frame in observation_list:
            all_rgb.append(np.array(Image.fromarray(frame.astype(np.uint8))))

        return all_rgb

    def on_valid_log(
        self,
        metrics: Dict[str, Any],
        metric_means: Dict[str, float],
        checkpoint_file_name: str,
        tasks_data: List[Any],
        scalar_name_to_total_experiences_key: Dict[str, str],
        step: int,
        **kwargs,
    ) -> None:
        """Log the validation metrics to wandb."""

        self._define_missing_metrics(
            metric_means=metric_means,
            scalar_name_to_total_experiences_key=scalar_name_to_total_experiences_key,
        )

        wandb.log(
            {
                **metric_means,
                "training_step": step,
            }
        )

    def get_table_content(self, metrics, tasks_data, frames_with_logit_flag=False):
        observation_list = []
        path_list = []
        frames_with_logits_list_numpy = []
        tasks_list = []
        for i in range(len(tasks_data)):
            if tasks_data[i]["local_logging_callback_sensor"] is not None:
                observation_list.append(
                    tasks_data[i]["local_logging_callback_sensor"]["observations"]
                )
                path_list.append(
                    tasks_data[i]["local_logging_callback_sensor"]["path"]
                )
                tasks_list.append(metrics["tasks"][i])
                if frames_with_logit_flag:
                    frames_with_logits_list_numpy.append(
                        tasks_data[i]["local_logging_callback_sensor"]["frames_with_logits"]
                    )

        list_of_video_frames = [
            self.combine_rgb_across_episode(obs) for obs in observation_list
        ]

        table_content = []

        for idx, data in enumerate(
            zip(list_of_video_frames, path_list, tasks_list)
        ):
            frames_without_logits, path, metric_data = (
                data[0],
                data[1],
                data[2],
            )
            wandb_data = [
                wandb.Video(
                    np.moveaxis(
                        np.array(frames_without_logits), [0, 3, 1, 2], [0, 1, 2, 3]
                    ),
                    fps=10,
                    format="mp4",
                ),
            ]
            if frames_with_logit_flag:
                frames_with_logits = frames_with_logits_list_numpy[idx]
                wandb_data.append(
                    wandb.Video(
                        np.moveaxis(
                            np.array(frames_with_logits), [0, 3, 1, 2], [0, 1, 2, 3]
                        ),
                        fps=10,
                        format="mp4",
                    )
                )

            wandb_data += [
                wandb.Image(path[0]),
                metric_data["ep_length"],
                metric_data["success"],
                metric_data["dist_to_target"],
                metric_data["task_info"]["sceneDataset"],
                metric_data["task_info"]["house_name"],
                metric_data["task_info"]["object_type"],
                metric_data["task_info"]["id"],
                idx,
            ]
            wandb_data = tuple(wandb_data)

            table_content.append(wandb_data)

        return table_content

    def on_test_log(
        self,
        checkpoint_file_name: str,
        metrics: Dict[str, Any],
        metric_means: Dict[str, float],
        tasks_data: List[Any],
        scalar_name_to_total_experiences_key: Dict[str, str],
        step: int,
        **kwargs,
    ) -> None:
        """Log the test metrics to wandb."""

        self._define_missing_metrics(
            metric_means=metric_means,
            scalar_name_to_total_experiences_key=scalar_name_to_total_experiences_key,
        )

        if "local_logging_callback_sensor" in tasks_data[0].keys():
            if tasks_data[0]["local_logging_callback_sensor"] is None:
                wandb.log(
                    {
                        **metric_means,
                        "training_step": step,
                    }
                )
            else:
                frames_with_logits_flag = False

                if "frames_with_logits" in tasks_data[0]["local_logging_callback_sensor"]:
                    frames_with_logits_flag = True

                table_content = self.get_table_content(metrics, tasks_data, frames_with_logits_flag)

                table = wandb.Table(
                    columns=[
                        "Trajectory",
                        "Trajectory with logits",
                        "Path",
                        "Episode Length",
                        "Success",
                        "Dist to target",
                        "Task Type",
                        "House Name",
                        "Target Object Type",
                        "Task Id",
                        "Index",
                    ]
                )

                for data in table_content:
                    table.add_data(*data)

                # TODO: Add with logit videos separately

                wandb.log(
                    {
                        **metric_means,
                        "training_step": step,
                        "Qualitative Examples": table,
                    }
                )
        else:
            wandb.log(
                {
                    **metric_means,
                    "training_step": step,
                }
            )

    def after_save_project_state(self, base_dir: str) -> None:
        pass

    def callback_sensors(self) -> Optional[Sequence[Sensor]]:
        return [
            LocalLoggingSensor(
                uuid="local_logging_callback_sensor", observation_space=gym.spaces.Discrete(1)
            ),
        ]
