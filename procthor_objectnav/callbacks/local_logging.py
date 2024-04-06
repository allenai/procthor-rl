import glob
import json
import os
from typing import Any, Dict, List, Optional, Sequence
from typing_extensions import Literal

import numpy as np
import wandb
from allenact.base_abstractions.callbacks import Callback
from allenact.base_abstractions.sensor import Sensor
import gym

import os
from typing import Dict, Any, Optional, List
import json
from moviepy.editor import ImageSequenceClip
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from ai2thor.controller import Controller

from allenact.base_abstractions.sensor import Sensor

from procthor_objectnav.tasks.object_nav import ProcTHORObjectNavTask


def get_top_down_path_view(controller, agent_path):
    if len(controller.last_event.third_party_camera_frames) < 2:
        event = controller.step({"action": "GetMapViewCameraProperties"})
        cam = event.metadata["actionReturn"].copy()
        cam["orthographicSize"] += 1
        controller.step(
            {"action": "AddThirdPartyCamera", "skyboxColor": "white", **cam}
        )

    # put this over the waypoints just in case
    event = controller.step({"action": "VisualizePath", "positions": agent_path})
    controller.step({"action": "HideVisualizedPath"})
    path = event.third_party_camera_frames[-1]
    return path


def unnormalize_image(img):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img


class LocalLoggingSensor(Sensor[Controller, ProcTHORObjectNavTask]):
    def get_observation(
        self, env: Controller, task: ProcTHORObjectNavTask, *args: Any, **kwargs: Any
    ) -> Any:
        # TODO make sure get observation is being called in the right place
        # TODO: set abstract task property visualize, true only if mode is eval
        if not task.visualize:
            return None

        # NOTE: Create top-down trajectory path visualization
        agent_path = [
            dict(x=p["x"], y=0.25, z=p["z"])
            for p in task._metrics["task_info"]["followed_path"]
        ]

        path = get_top_down_path_view(env, agent_path)

        # this is the connection to the task id sensor
        if os.getenv("LOGGING_DIR") is None:
            d = "output"
        else:
            d = os.getenv("LOGGING_DIR")
        df = pd.read_csv(
            f"{d}/ac-data/{task.task_info['id']}.txt",
            names=list(task.class_action_names()) + ["EstimatedValue"],  #
        )

        # TODO: ep length exists? maybe should be basic/shared metric
        ep_length = task._metrics["ep_length"]

        # get returns from each step
        returns = []

        if "rewards" in task.task_info:
            # TODO: rewards exists? keep machinery with dummy value for later?
            for r in reversed(task.task_info["rewards"]):
                if len(returns) == 0:
                    returns.append(r)
                else:
                    returns.append(r + returns[-1] * 0.99)  # gamma value
            returns = returns[::-1]
        else:
            returns = [None for x in task.task_info["action_successes"]]

        video_frames = []
        for step in range(task._metrics["ep_length"] + 1):
            is_first_frame = step == 0
            is_last_frame = step == task._metrics["ep_length"]

            # TODO change this to get the observation history - need both cameras and size
            # todo fix width to be 2*224 (or something else)

            if task.observations[step].dtype == np.uint8:
                unnormalized_frame = task.observations[step]
                agent_frame = np.array(Image.fromarray(unnormalized_frame))
            else:
                unnormalized_frame = unnormalize_image(
                    task.observations[step]
                )
                agent_frame = np.array(
                    Image.fromarray((unnormalized_frame * 255).astype(np.uint8)).resize(
                        (224, 224)
                    )
                )

            frame_number = step
            # todo: remove? since doesn't apply to all tasks
            if "dist_to_target" in task.task_info:
                dist_to_target = task.task_info["dist_to_target"][step]
            else:
                dist_to_target = np.inf

            if is_first_frame:
                last_action_success = None
                last_reward = None
                return_value = None
            else:
                last_action_success = task.task_info["action_successes"][step - 1]
                if "rewards" in task.task_info:
                    last_reward = task.task_info["rewards"][step - 1]
                    return_value = returns[step - 1]
                else:
                    last_reward = None
                    return_value = None

            if is_last_frame:
                action_dist = None
                critic_value = None
                taken_action = None
            else:
                policy_critic_value = df.iloc[step].values.tolist()
                action_dist = policy_critic_value[: len(task.class_action_names())]
                critic_value = policy_critic_value[-1]

                taken_action = task.task_info["taken_actions"][step]

            video_frame = self.get_video_frame(
                agent_frame=agent_frame,
                frame_number=frame_number,
                action_names=list(task.class_action_names()),
                last_reward=(
                    round(last_reward, 2) if last_reward is not None else None
                ),
                critic_value=(
                    round(critic_value, 2) if critic_value is not None else None
                ),
                return_value=(
                    round(return_value, 2) if return_value is not None else None
                ),
                dist_to_target=round(dist_to_target, 2),
                action_dist=action_dist,
                ep_length=ep_length,
                last_action_success=last_action_success,
                taken_action=taken_action,
            )
            video_frames.append(video_frame)

        # TODO: add one more directory for check

        task_success = "Success" if task._metrics["success"] else "Failure"

        traj_info_dir = f"{d}/trajectories/{task_success}/{task.task_info['id']}"

        os.makedirs(traj_info_dir, exist_ok=True)

        # TODO: create a success and failure directory
        # os.makedirs(os.path.join(traj_info_dir, "Success"), exist_ok=True)
        # os.makedirs(os.path.join(traj_info_dir, "Failure"), exist_ok=True)

        imsn = ImageSequenceClip([frame for frame in video_frames], fps=10)
        imsn.write_videofile(os.path.join(traj_info_dir, "frames.mp4"), fps=10)
        # imsn.write_videofile(f"output/trajectories/{task.task_info['id']}/frames.mp4")

        # save the top-down path
        Image.fromarray(path).save(os.path.join(traj_info_dir, "path.png"))
        # Image.fromarray(path).save(f"output/trajectories/{task.task_info['id']}/path.png")

        # TODO: keep?
        # save the value function over time
        fig, ax = plt.subplots()
        estimated_values = df.EstimatedValue.to_numpy()
        ax.plot(estimated_values, label="Critic Estimated Value")
        ax.plot(returns, label="Return")
        ax.set_ylabel("Value")
        ax.set_xlabel("Time Step")
        ax.set_title("Value Function over Time")
        ax.legend()
        fig.savefig(
            os.path.join(traj_info_dir, "value_fn.svg"),
            bbox_inches="tight",
        )
        # fig.savefig(
        #     f"output/trajectories/{task.task_info['id']}/value_fn.svg",
        #     bbox_inches="tight",
        # )
        plt.clf()

        # TODO: rewrite to be flexible for multi-task
        task_out = {
            "id": task.task_info["id"],
            "task_type": task.task_info["sceneDataset"],
            "spl": task._metrics["spl"],
            "success": task._metrics["success"],
            "finalDistance": task.task_info["dist_to_target"][-1],
            "initialDistance": task.task_info["dist_to_target"][0],
            "minDistance": min(task.task_info["dist_to_target"]),
            "episodeLength": task._metrics["ep_length"],
            "confidence": (
                None
                if task.task_info["taken_actions"][-1] != "End"
                else df.End.to_list()[-1]
            ),
            "failedActions": len(
                [s for s in task.task_info["action_successes"] if not s]
            ),
            "targetObjectType": task.task_info["object_type"],
            "numTargetObjects": len(task.task_info["target_object_ids"]),
            "mirrored": task.task_info["mirrored"],
            "scene": {
                "name": task.task_info["house_name"],
                "split": task.task_info["mode"],
                "rooms": 1,
            },
        }

        with open(os.path.join(traj_info_dir, "data.json"), "w") as f:
            # with open(f"output/trajectories/{task.task_info['id']}/data.json", "w") as f:
            json.dump(
                task_out,
                f,
            )

        return {
            "observations": task.observations,
            "path": [path],  # path,
            "frames_with_logits": video_frames,
            **task._metrics,
        }

    @staticmethod
    def get_mapping_video_frame(
        aggregate_map: np.ndarray,
        frame_number: int,
    ) -> np.array:
        agent_height, agent_width, ch = aggregate_map.shape
        try:
            font_to_use = "./Arial.ttf"  # possibly need a full path here
            full_font_load = ImageFont.truetype(font_to_use, 8)
        except:
            font_to_use = "./LiberationSans-Regular.ttf"
            full_font_load = ImageFont.truetype(font_to_use, 8)

        IMAGE_BORDER = 3
        image_dims = (
            agent_height + 2 * IMAGE_BORDER,
            agent_width + 2 * IMAGE_BORDER,
            ch,
        )
        image = np.full(image_dims, 255, dtype=np.uint8)

        image[
            IMAGE_BORDER : IMAGE_BORDER + agent_height,
            IMAGE_BORDER : IMAGE_BORDER + agent_width,
            :,
        ] = (
            aggregate_map * 255
        )

        text_image = Image.fromarray(image)
        img_draw = ImageDraw.Draw(text_image)

        img_draw.text(
            (IMAGE_BORDER * 1.1, IMAGE_BORDER * 1),
            str(frame_number),
            font=full_font_load,  # ImageFont.truetype(font_to_use, 25),
            fill="black",
        )

        return np.array(text_image)

    @staticmethod
    def get_video_frame(
        agent_frame: np.ndarray,
        frame_number: int,
        action_names: List[str],
        last_reward: Optional[float],
        critic_value: Optional[float],
        return_value: Optional[float],
        dist_to_target: float,
        action_dist: Optional[List[float]],
        ep_length: int,
        last_action_success: Optional[bool],
        taken_action: Optional[str],
    ) -> np.array:
        agent_height, agent_width, ch = agent_frame.shape

        try:
            font_to_use = "./Arial.ttf"  # possibly need a full path here
            full_font_load = ImageFont.truetype(font_to_use, 8)
        except:
            font_to_use = "./LiberationSans-Regular.ttf"
            full_font_load = ImageFont.truetype(font_to_use, 8)

        IMAGE_BORDER = 25
        TEXT_OFFSET_H = 60
        TEXT_OFFSET_V = 30

        image_dims = (
            agent_height + 2 * IMAGE_BORDER + 30,
            agent_width + 2 * IMAGE_BORDER + 400,
            ch,
        )
        image = np.full(image_dims, 255, dtype=np.uint8)

        image[
            IMAGE_BORDER : IMAGE_BORDER + agent_height,
            IMAGE_BORDER : IMAGE_BORDER + agent_width,
            :,
        ] = agent_frame

        text_image = Image.fromarray(image)
        img_draw = ImageDraw.Draw(text_image)
        # font size 25, aligned center and middle
        if action_dist is not None:
            for i, (prob, action) in enumerate(zip(action_dist, action_names)):
                if i < 10:
                    img_draw.text(
                        (
                            IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H,
                            (TEXT_OFFSET_V + 5) + i * 10,
                        ),
                        action,
                        font=ImageFont.truetype(font_to_use, 10),
                        fill="gray" if action != taken_action else "black",
                        anchor="rm",
                    )
                    img_draw.rectangle(
                        (
                            IMAGE_BORDER * 2 + agent_width + (TEXT_OFFSET_H + 5),
                            TEXT_OFFSET_V + i * 10,
                            IMAGE_BORDER * 2
                            + agent_width
                            + (TEXT_OFFSET_H + 5)
                            + int(100 * prob),
                            (TEXT_OFFSET_V + 5) + i * 10,
                        ),
                        outline="blue",
                        fill="blue",
                    )
                else:
                    img_draw.text(
                        (
                            IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H + 200,
                            (TEXT_OFFSET_V + 5) + (i - 10) * 10,
                        ),
                        action,
                        font=ImageFont.truetype(font_to_use, 10),
                        fill="gray" if action != taken_action else "black",
                        anchor="rm",
                    )
                    img_draw.rectangle(
                        (
                            IMAGE_BORDER * 2 + agent_width + (TEXT_OFFSET_H + 205),
                            TEXT_OFFSET_V + (i - 10) * 10,
                            IMAGE_BORDER * 2
                            + agent_width
                            + (TEXT_OFFSET_H + 205)
                            + int(100 * prob),
                            (TEXT_OFFSET_V + 5) + (i - 10) * 10,
                        ),
                        outline="blue",
                        fill="blue",
                    )

        img_draw.text(
            (IMAGE_BORDER * 1.1, IMAGE_BORDER * 1),
            str(frame_number),
            font=full_font_load,  # ImageFont.truetype(font_to_use, 25),
            fill="white",
        )

        oset = -10
        if last_reward is not None:
            img_draw.text(
                (
                    IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H,
                    IMAGE_BORDER * 1 + 175 + oset,
                ),
                "Last Reward:",
                font=full_font_load,  # ImageFont.truetype(font_to_use, 14),
                fill="gray",
                anchor="rm",
            )
            img_draw.text(
                (
                    IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H,
                    IMAGE_BORDER * 1 + 175 + oset,
                ),
                " " + ("+" if last_reward > 0 else "") + str(last_reward),
                font=full_font_load,  # ImageFont.truetype(font_to_use, 14),
                fill="gray",
                anchor="lm",
            )

        oset = 10
        if critic_value is not None:
            img_draw.text(
                (
                    IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H,
                    IMAGE_BORDER * 1 + 175 + oset,
                ),
                "Critic Value:",
                font=full_font_load,  # ImageFont.truetype(font_to_use, 14),
                fill="gray",
                anchor="rm",
            )
            img_draw.text(
                (
                    IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H,
                    IMAGE_BORDER * 1 + 175 + oset,
                ),
                " " + ("+" if critic_value > 0 else "") + str(critic_value),
                font=full_font_load,  # ImageFont.truetype(font_to_use, 14),
                fill="gray",
                anchor="lm",
            )

        if return_value is not None:
            img_draw.text(
                (
                    IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H,
                    IMAGE_BORDER * 1 + 195 + oset,
                ),
                "Return:",
                font=full_font_load,  # ImageFont.truetype(font_to_use, 14),
                fill="gray",
                anchor="rm",
            )
            img_draw.text(
                (
                    IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H,
                    IMAGE_BORDER * 1 + 195 + oset,
                ),
                " " + ("+" if return_value > 0 else "") + str(return_value),
                font=full_font_load,  # ImageFont.truetype(font_to_use, 14),
                fill="gray",
                anchor="lm",
            )

        if last_action_success is not None:
            img_draw.text(
                (
                    IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H,
                    IMAGE_BORDER * 1 + 235,
                ),
                "Last Action:",
                font=full_font_load,  # ImageFont.truetype(font_to_use, 14),
                fill="gray",
                anchor="rm",
            )
            img_draw.text(
                (
                    IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H,
                    IMAGE_BORDER * 1 + 235,
                ),
                " Success" if last_action_success else " Failure",
                font=full_font_load,  # ImageFont.truetype(font_to_use, 14),
                fill="green" if last_action_success else "red",
                anchor="lm",
            )

        if taken_action == "manual override":
            img_draw.text(
                (
                    IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H + 50,
                    TEXT_OFFSET_V + 5 * 20,
                ),
                "Manual Override",
                font=full_font_load,  # ImageFont.truetype(font_to_use, 14),
                fill="red",
                anchor="rm",
            )

        img_draw.text(
            (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H, IMAGE_BORDER * 1 + 145),
            "Target Dist:",
            font=full_font_load,  # ImageFont.truetype(font_to_use, 14),
            fill="gray",
            anchor="rm",
        )
        img_draw.text(
            (IMAGE_BORDER * 2 + agent_width + TEXT_OFFSET_H, IMAGE_BORDER * 1 + 145),
            f" {dist_to_target}m",
            font=full_font_load,  # ImageFont.truetype(font_to_use, 14),
            fill="gray",
            anchor="lm",
        )

        lower_offset = 10
        progress_bar_height = 20

        img_draw.rectangle(
            (
                IMAGE_BORDER,
                agent_height + IMAGE_BORDER + lower_offset,
                IMAGE_BORDER + agent_width,
                agent_height + IMAGE_BORDER + progress_bar_height + lower_offset,
            ),
            outline="lightgray",
            fill="lightgray",
        )
        img_draw.rectangle(
            (
                IMAGE_BORDER,
                agent_height + IMAGE_BORDER + lower_offset,
                IMAGE_BORDER + int(frame_number * agent_width / ep_length),
                agent_height + IMAGE_BORDER + progress_bar_height + lower_offset,
            ),
            outline="blue",
            fill="blue",
        )

        return np.array(text_image)


class WandbLoggingSensor(LocalLoggingSensor):
    """
    Create a wandb logging sensor that just calculates things
    and returns to wandb logging instead of saving it locally
    """

    def get_observation(
        self, env: Controller, task: ProcTHORObjectNavTask, *args: Any, **kwargs: Any
    ) -> Any:
        if not task.visualize:
            return None

        # NOTE: Create top-down trajectory path visualization
        agent_path = [
            dict(x=p["x"], y=0.25, z=p["z"])
            for p in task._metrics["task_info"]["followed_path"]
        ]

        path = get_top_down_path_view(env, agent_path)

        return {
            "observations": task.observations,
            "path": [path],  # path,
            # "frames_with_logits": video_frames,
            **task._metrics,
        }
