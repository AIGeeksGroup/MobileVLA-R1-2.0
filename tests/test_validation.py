"""Small regression cases; no model weights, network, or robot hardware required."""
import unittest

from mobilevla_r1.controllers import CallbackController
from mobilevla_r1.evaluation import navigation_metrics
from mobilevla_r1.schema import TaskAction, structured_parts


class NavigationLogTests(unittest.TestCase):
    def episode(self, **changes):
        return {"final_distance": 1.0, "oracle_distance": 0.5,
                "shortest_path_length": 5.0, "traveled_distance": 10.0,
                "stopped": True, **changes}

    def test_path_efficiency_and_explicit_stop(self):
        result = navigation_metrics({"a": self.episode(), "b": self.episode(stopped=False)})
        self.assertEqual(result["SR"], 0.5)
        self.assertEqual(result["SPL"], 0.25)
        self.assertEqual(result["OS"], 1.0)

    def test_string_false_cannot_count_as_success(self):
        with self.assertRaises(ValueError):
            navigation_metrics({"a": self.episode(stopped="false")})

    def test_nonfinite_metric_inputs_are_rejected(self):
        for changes in ({"success_distance": float("nan")},
                        {"final_distance": float("inf")},
                        {"dtw_distance": float("nan"), "reference_path_points": 3}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                navigation_metrics({"a": self.episode(**changes)})


class ActionValidationTests(unittest.TestCase):
    def test_nan_limits_do_not_disable_controller_bounds(self):
        with self.assertRaises(ValueError):
            CallbackController(lambda *args: None, lambda: None, {},
                               velocity_limits=(float("nan"), 1.0, 1.0))

    def test_stop_does_not_dispatch_motion_even_with_large_velocity(self):
        events = []
        controller = CallbackController(lambda *args: events.append("move"),
                                        lambda: events.append("stop"), {})
        controller.execute(TaskAction(100.0, 0.0, 0.0, "stop"))
        self.assertEqual(events, ["stop"])

    def test_unknown_behavior_stops_and_fails(self):
        events = []
        controller = CallbackController(lambda *args: events.append("move"),
                                        lambda: events.append("stop"), {})
        with self.assertRaises(ValueError):
            controller.execute(TaskAction(0.0, 0.0, 0.0, "unknown"))
        self.assertEqual(events, ["stop"])

    def test_bool_is_not_a_velocity(self):
        with self.assertRaises(ValueError):
            TaskAction(True, 0.0, 0.0, "locomotion")

    def test_format_rejects_nontext_and_duplicate_tags(self):
        self.assertIsNone(structured_parts(None))
        self.assertIsNone(structured_parts("<think>a</think><answer>b</answer><answer>c</answer>"))
        self.assertEqual(structured_parts("<think>a</think><answer>b</answer>"), ("a", "b"))


if __name__ == "__main__":
    unittest.main()
