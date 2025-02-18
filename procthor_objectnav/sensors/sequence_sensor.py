import random

import gym
import numpy as np
from allenact.base_abstractions.sensor import Sensor
from allenact.utils.misc_utils import prepare_locals_for_super


class TimeStepSensor(Sensor):
    def __init__(self, uuid: str = "time_step", max_time_for_random_shift=0) -> None:
        observation_space = self._get_observation_space()
        self.max_time_for_random_shift = max_time_for_random_shift
        self.random_start = 0
        super().__init__(**prepare_locals_for_super(locals()))
        self._update = False

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(1)

    def sample_random_start(self):
        self.random_start = random.randint(0, max(self.max_time_for_random_shift, 0))

    def get_observation(  # type:ignore
        self,
        env,
        task,
        *args,
        **kwargs,
    ) -> np.ndarray:
        steps = task.num_steps_taken()
        if self._update:
            steps += 1
        else:
            self._update = True
        if task.is_done():  # not increment at next episode start
            self._update = False
            self.sample_random_start()
        return np.array(self.random_start + int(steps), dtype=np.int64)


class TrajectorySensor(Sensor):
    def __init__(self, uuid: str = "traj_index", max_idx: int = 128) -> None:
        observation_space = self._get_observation_space()
        self.curr_idx = 0
        self.max_idx = max_idx
        super().__init__(**prepare_locals_for_super(locals()))
        self._update = False

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(1)

    def get_observation(  # type:ignore
        self,
        env,
        task,
        *args,
        **kwargs,
    ) -> np.ndarray:
        # if task.num_steps_taken() == 0:
        if self._update:
            self.curr_idx += 1
            if self.curr_idx >= self.max_idx:
                self.curr_idx = 0
            self._update = False
        if task.is_done():  # update at next episode start
            self._update = True
        return np.array(self.curr_idx, dtype=np.int64)
