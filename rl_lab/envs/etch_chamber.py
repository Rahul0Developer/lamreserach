"""EtchChamberEnv: semiconductor-flavoured control problem.

Physics sketch (deliberately simple, but mass-balance honest):
  - Chamber pressure P [Torr] responds to pump throttle position u_p:
        dP = k_p * (u_p - P) * dt          (first-order lag toward setpoint)
  - Etch rate R [A/min] follows a Langmuir-Hinshelwood-like law:
        R = k_e * Q / (1 + Q / Q_sat) * f(P)   with f(P) peaking at P_opt
    so gas flow Q and pressure interact -- you cannot max both and win.
  - The "process" is the error e = R - target. Reward penalises |e| plus
    actuator effort (changing valves fast wears hardware in real tools).

State for tabular agents is bucketed: (pressure level, etch-rate error sign,
flow level). Continuous physics underneath, discrete view on top -- same
split real equipment software does between process sim and recipe logic.

Noise: sensor readings get Gaussian noise, like a real pressure gauge.
Seeded via numpy Generator so runs are reproducible (seed control matters
in RL more than anywhere else).
"""
from __future__ import annotations

import numpy as np

from rl_lab.core.base import BaseEnvironment, StepResult

# Action table: index -> (delta_pressure_setpoint, delta_gas_flow).
# Three coarsely-spaced knobs per axis keeps the tabular state space small
# enough for Q-learning to actually converge in a few hundred episodes.
ACTIONS = [
    (-1.0, -1.0), (-1.0, 0.0), (-1.0, +1.0),
    ( 0.0, -1.0), ( 0.0, 0.0), ( 0.0, +1.0),
    (+1.0, -1.0), (+1.0, 0.0), (+1.0, +1.0),
]

# Pressure levels (Torr) we bucket into; error sign is its own axis.
PRESSURE_BINS = (5.0, 15.0)          # low / mid / high
ERROR_BINS = (-2.0, 2.0)             # under / on-target / over (A/min)
FLOW_BINS = (40.0, 80.0)             # low / mid / high sccm


class EtchChamberEnv(BaseEnvironment):
    name = "etch_chamber"

    def __init__(self, target_etch_rate: float = 50.0,
                 process_time_s: float = 60.0,
                 sensor_noise: float = 1.0,
                 seed: int | None = None) -> None:
        self.target = float(target_etch_rate)
        self.process_time = float(process_time_s)
        self.sensor_noise = float(sensor_noise)
        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._reset_state()

    # -- internal model -----------------------------------------------------

    def _reset_state(self) -> None:
        # Start slightly off-recipe: a cold chamber at ambient-ish pressure.
        self.pressure = 20.0            # Torr
        self.p_setpoint = 20.0
        self.gas_flow = 25.0            # sccm
        self.etch_rate = 0.0
        self._step_count = 0

    @staticmethod
    def _pressure_response(p: float) -> float:
        """Multiplicative factor on etch rate; peaks around 10 Torr."""
        return float(np.exp(-((p - 10.0) ** 2) / (2 * 8.0 ** 2)))

    def _true_etch_rate(self) -> float:
        k_e, q_sat = 90.0, 120.0
        return k_e * self.gas_flow / (self.gas_flow + q_sat) * self._pressure_response(self.pressure) * 2.2

    # -- BaseEnvironment API --------------------------------------------------

    def reset(self, *, seed: int | None = None) -> tuple[tuple, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._reset_state()
        return self._observe(), {"target": self.target}

    def step(self, action: int) -> StepResult:
        if not isinstance(action, (int, np.integer)) or not 0 <= action < len(ACTIONS):
            raise ValueError(f"invalid action {action!r}, expected 0..{len(ACTIONS)-1}")

        dp, dq = ACTIONS[int(action)]
        dt = 1.0
        # Valves move gradually; setpoints change by the action increment.
        self.p_setpoint = float(np.clip(self.p_setpoint + dp * 2.0, 1.0, 30.0))
        self.gas_flow = float(np.clip(self.gas_flow + dq * 10.0, 0.0, 150.0))

        # First-order lag: chamber pressure relaxes toward the setpoint.
        k_p = 0.35
        self.pressure += k_p * (self.p_setpoint - self.pressure) * dt

        true_rate = self._true_etch_rate()
        # Small process noise on the actual rate too (plasma instability).
        self.etch_rate = true_rate + self._rng.normal(0.0, 0.5)

        measured_error = self.etch_rate - self.target + \
            self._rng.normal(0.0, self.sensor_noise)

        reward = -abs(measured_error) / self.target  # normalised: -1 .. ~0
        # Actuator-effort penalty: real valves wear out; discourages chatter.
        reward -= 0.01 * (abs(dp) + abs(dq))

        self._step_count += 1
        truncated = self._step_count * dt >= self.process_time
        info = {"etch_rate": self.etch_rate, "pressure": self.pressure,
                "gas_flow": self.gas_flow, "error": measured_error}
        return StepResult(observation=self._observe(), reward=float(reward),
                          terminated=False, truncated=bool(truncated), info=info)

    def _observe(self) -> tuple:
        """Bucketed observation: (pressure_level, error_sign, flow_level)."""
        p_lvl = int(np.digitize(self.pressure, PRESSURE_BINS))
        err = self.etch_rate - self.target
        e_lvl = int(np.digitize(err, ERROR_BINS))
        f_lvl = int(np.digitize(self.gas_flow, FLOW_BINS))
        return (p_lvl, e_lvl, f_lvl)

    def action_space_size(self) -> int:
        return len(ACTIONS)

    def observation_space_size(self) -> int:
        return 3 * 3 * 3  # 3 bins on each of 3 axes
