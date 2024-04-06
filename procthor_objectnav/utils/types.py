from typing import Any, Dict, List, Optional, Tuple

try:
    from typing import Literal, TypedDict
except ImportError:
    from typing_extensions import Literal, TypedDict

from allenact.base_abstractions.sensor import Sensor
from attrs import define


class Vector3(TypedDict):
    x: float
    y: float
    z: float


@define
class TaskSamplerArgs:
    process_ind: int
    """The process index number."""

    mode: Literal["train", "eval"]
    """Whether we are in training or evaluation mode."""

    house_inds: List[int]
    """Which houses to use for each process."""

    houses: Any
    """The hugging face Dataset of all the houses in the split."""

    sensors: List[Sensor]
    """The sensors to use for each task."""

    controller_args: Dict[str, Any]
    """The arguments to pass to the AI2-THOR controller."""

    reward_config: Dict[str, Any]
    """The reward configuration to use."""

    target_object_types: List[str]
    """The object types to use as targets."""

    max_steps: int
    """The maximum number of steps to run each task."""

    max_tasks: int
    """The maximum number of tasks to run."""

    distance_type: str
    """The type of distance computation to use ("l2" or "geo")."""

    resample_same_scene_freq: int
    """
    Number of times to sample a scene/house before moving to the next one.
    
    If <1 then will never 
        sample a new scene (unless `force_advance_scene=True` is passed to `next_task(...)`.
    ."""

    p_randomize_materials: float
    test_on_validation: bool
    actions: Tuple[str]
    max_agent_positions: int
    max_vis_points: int
    p_greedy_target_object: float
    ithor_p_shuffle_objects: float
    valid_agent_heights: float
    visualize: bool = False

    # Can we remove?
    deterministic_cudnn: bool = False
    loop_dataset: bool = True
    seed: Optional[int] = None
    allow_flipping: bool = False


@define
class RewardConfig:
    step_penalty: float
    goal_success_reward: float
    failed_stop_reward: float
    shaping_weight: float
    reached_horizon_reward: float
    positive_only_reward: bool


class AgentPose(TypedDict):
    position: Vector3
    rotation: Vector3
    horizon: int
    standing: bool
