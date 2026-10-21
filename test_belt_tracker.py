import unittest

import numpy as np

import belt_tracker


class BeltTrackerTest(unittest.TestCase):
    def setUp(self):
        belt_tracker.reset()

    def test_round_trip_between_pixels_and_millimetres(self):
        millimetres = np.array([[120.0, 80.0], [400.0, 250.0]], np.float32)
        back = belt_tracker.to_belt_mm(belt_tracker.to_pixels(millimetres))
        np.testing.assert_allclose(back, millimetres, atol=0.05)

    def test_speed_and_pick_time_on_a_steady_belt(self):
        speed = 250.0
        x0, y_mm = 100.0, 80.0
        dt = 1 / 30
        rng = np.random.default_rng(0)
        last_t = 0.0
        for step in range(90):
            last_t = step * dt
            noisy = np.array([[x0 + speed * last_t, y_mm]]) + rng.normal(0, 0.2, size=(1, 2))
            belt_tracker.update(last_t, belt_tracker.to_pixels(noisy))

        self.assertIsNotNone(belt_tracker.belt_speed)
        self.assertAlmostEqual(belt_tracker.belt_speed, speed, delta=0.3)

        plan = belt_tracker.pick_plan()
        self.assertEqual(set(plan), {1})
        true_arrival = (belt_tracker.PICK_LINE_MM - x0) / speed
        self.assertAlmostEqual(plan[1]["t_pick"], true_arrival, delta=0.005)
        predicted = belt_tracker.predict_position(1, last_t + 0.5)
        self.assertAlmostEqual(predicted[0], x0 + speed * (last_t + 0.5), delta=1.0)

    def test_dropped_frame_keeps_the_track_and_a_blip_is_not_trusted(self):
        speed = 200.0
        dt = 1 / 30
        for step in range(20):
            if step == 8:
                continue
            t = step * dt
            centre = belt_tracker.to_pixels([[50 + speed * t, 100.0]])[0]
            extra = [centre]
            if step == 12:
                extra.append(belt_tracker.to_pixels([[10.0, 10.0]])[0])
            belt_tracker.update(t, np.array(extra, np.float32))
        self.assertEqual(len(belt_tracker.pick_plan()), 1)
        self.assertEqual(list(belt_tracker.history), [1])

    def _speed_error(self, duration, sigma):
        belt_tracker.reset()
        speed = 180.0
        dt = 1 / 30
        rng = np.random.default_rng(1)
        steps = int(duration / dt)
        for step in range(steps):
            t = step * dt
            noisy = np.array([[50 + speed * t, 120.0]]) + rng.normal(0, sigma, size=(1, 2))
            belt_tracker.update(t, belt_tracker.to_pixels(noisy))
        return abs(belt_tracker.belt_speed - speed)

    def test_a_longer_look_tightens_the_estimate(self):
        early = self._speed_error(0.5, 4.0)
        late = self._speed_error(4.0, 4.0)
        self.assertLess(late, early)
        self.assertLess(late, 1.0)

    def test_one_prediction_keeps_moving_for_three_seconds(self):
        speed = 200.0
        dt = 1 / 30
        last_t = 0.0
        for step in range(40):
            last_t = step * dt
            here = belt_tracker.to_pixels([[30 + speed * last_t, 90.0]])[0]
            other = belt_tracker.to_pixels([[30 + speed * last_t, 320.0]])[0]
            belt_tracker.update(last_t, np.array([here, other], np.float32))
        self.assertEqual(len(belt_tracker.history), 1)

        coast_t = last_t + 2.5
        belt_tracker.update(coast_t, np.zeros((0, 2), np.float32))
        predicted = belt_tracker.predict_position(1, coast_t)
        self.assertAlmostEqual(predicted[0], 30 + speed * coast_t, delta=1.0)
        self.assertEqual(len(belt_tracker.snapshot(coast_t)["tracks"][0]["points"]), 1)

        belt_tracker.update(last_t + 3.05, np.zeros((0, 2), np.float32))
        self.assertEqual(belt_tracker.history, {})
        self.assertIsNone(belt_tracker.predict_position(1, last_t + 3.05))

    def test_prediction_follows_the_moving_knife_not_a_static_one(self):
        speed = 220.0
        dt = 1 / 20
        last_t = 0.0
        for step in range(30):
            last_t = step * dt
            static = belt_tracker.to_pixels([[80.0, 40.0]])[0]
            moving = belt_tracker.to_pixels([[20 + speed * last_t, 200.0]])[0]
            belt_tracker.update(last_t, np.array([static, moving], np.float32))
        samples = next(iter(belt_tracker.history.values()))
        self.assertAlmostEqual(samples[-1][1], 20 + speed * last_t, delta=20)
        self.assertAlmostEqual(samples[-1][2], 200.0, delta=15)

        coast_t = last_t + 1.0
        belt_tracker.update(coast_t, np.array([belt_tracker.to_pixels([[80.0, 40.0]])[0]], np.float32))
        predicted = belt_tracker.predict_position(next(iter(belt_tracker.history)), coast_t)
        self.assertIsNotNone(predicted)
        self.assertAlmostEqual(predicted[0], 20 + speed * coast_t, delta=30)
        self.assertAlmostEqual(predicted[1], 200.0, delta=20)
