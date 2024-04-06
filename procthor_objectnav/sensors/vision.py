import numpy as np
from ai2thor.controller import Controller

from allenact.base_abstractions.task import Task
from allenact.embodiedai.sensors.vision_sensors import RGBSensor


class RGBSensorThorController(RGBSensor[Controller, Task[Controller]]):
    """Sensor for RGB images in THOR.

    Returns from a running instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(self, env: Controller, task: Task[Controller]) -> np.ndarray:
        return env.last_event.frame.copy()
