"""Task actions to embodiment-specific fixed routines (System 0).

Controller callbacks are injected by the deployment application. They must own
tracking, IK, contacts, completion detection and the robot-specific SDK connection.
"""
import math
from typing import Callable, Mapping, Protocol
from mobilevla_r1.schema import TaskAction


class RobotController(Protocol):
    def execute(self, action: TaskAction): ...
    def stop(self): ...


class CallbackController:
    def __init__(self, move: Callable, stop: Callable, routines: Mapping[str, Callable],
                 velocity_limits=(1.0, 1.0, 1.0), locomotion_behavior="locomotion", stop_behavior="stop"):
        self.move_callback = move
        self.stop_callback = stop
        self.routines = dict(routines)
        self.limits = velocity_limits
        if len(velocity_limits) != 3 or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in velocity_limits):
            raise ValueError("Three positive controller velocity limits required")
        self.locomotion_behavior = locomotion_behavior
        self.stop_behavior = stop_behavior

    def execute(self, action):
        if action.behavior == self.stop_behavior:
            return self.stop()
        velocity = (action.vx, action.vy, action.omega)
        if any(abs(value) > limit for value, limit in zip(velocity, self.limits)):
            self.stop()
            raise ValueError("Predicted velocity exceeds this controller's configured limits")
        if action.behavior == self.locomotion_behavior:
            return self.move_callback(*velocity)
        if action.behavior not in self.routines:
            self.stop()
            raise ValueError(f"No fixed controller routine registered for {action.behavior}")
        # Each fixed behavior routine receives the complete task action so it can
        # coordinate locomotion and manipulation, and must block until complete.
        return self.routines[action.behavior](action)

    def stop(self):
        return self.stop_callback()


def closed_loop(policy, observe, controller: RobotController, max_steps=100, stop_behavior="stop"):
    """observe() -> (record, loaded_observation); policy owns no robot SDK state."""
    from mobilevla_r1.inference import predict
    try:
        for _ in range(max_steps):
            record, observation = observe()
            result = predict(policy, record, observation)
            action = TaskAction(*result["action"]["velocity"], result["action"]["behavior"])
            controller.execute(action)
            if action.behavior == stop_behavior:
                return
    finally:
        controller.stop()
