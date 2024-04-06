from typing import Any
import gym
from allenact.base_abstractions.sensor import Sensor


class TaskIdSensor(Sensor):
    def __init__(
        self,
        uuid: str = "task_id_sensor",
        **kwargs: Any,
    ):
        super().__init__(uuid=uuid, observation_space=gym.spaces.Discrete(1))

    def _get_observation_space(self):
        return gym.spaces.Discrete(1000)

    def get_observation(
        self,
        env,
        task,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        out = [ord(k) for k in task.task_info["id"]]
        for _ in range(len(out), 1000):
            out.append(ord(" "))
        return out