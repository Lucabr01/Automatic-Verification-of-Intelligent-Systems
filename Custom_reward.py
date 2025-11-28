import numpy as np
from typing import List, Union
from sinergym.utils.rewards import BaseReward


class ExponentialThermalReward(BaseReward):
    """
    Exponential thermal reward for datacenter / HVAC control.

    General form:
        r_t = - w_E * energy_penalty - w_T * comfort_penalty

    where:
        energy_penalty  = E_t / energy_scale
        comfort_penalty = f_T(T_max)

    The thermal penalty f_T(T) is piecewise:

        - Comfort / cold (T <= T_high):
              f_T(T) = 0

        - Warning zone (T_high < T < T_red):
              f_T(T) = exp(alpha * (T - T_high)) - 1

        - Red zone (T >= T_red):
              f_T(T) = C_AL + exp(beta * (T - T_red)) - 1

    with:
        C_AL = exp(alpha * (T_red - T_high)) - 1

    Supports:
        - temp_name: str       -> single zone temperature
        - temp_name: list[str] -> multiple zones, uses max temperature
    """

    def __init__(
        self,
        w_E: float = 0.5,
        w_T: float = 1.0,
        alpha: float = 0.5,
        beta: float = 1.5,
        temp_name: Union[str, List[str]] = "Zone Air Temperature(SPACE1-1)",
        energy_name: str = "Facility Total HVAC Electricity Demand Rate(Whole Building)",
        energy_scale: float = 10_000.0,
        T_low: float = 20.0,
        T_high: float = 25.0,
        T_red: float = 28.0,
        max_exponent: float = 10.0,
    ):
        # Sinergym v3+: BaseReward without env
        super().__init__()

        # Weights
        self.w_E = w_E
        self.w_T = w_T
        self.alpha = alpha
        self.beta = beta

        # One or multiple temperature variables
        if isinstance(temp_name, str):
            self.temp_names = [temp_name]
        else:
            self.temp_names = list(temp_name)

        self.energy_name = energy_name
        self.energy_scale = energy_scale

        # Thermal thresholds
        self.T_low = T_low
        self.T_high = T_high
        self.T_red = T_red

        # Numerical stability
        self.max_exponent = max_exponent

        # Ensure continuity between warning and red zone at T_red
        warning_at_red = min(
            self.alpha * (self.T_red - self.T_high),
            self.max_exponent,
        )
        self.C_AL_calc = float(np.exp(warning_at_red) - 1.0)

        # Internal debug info (not used by Sinergym / WandB)
        self.last_info = {}

    # ------------------------------------------------------------------
    # Thermal penalty f_T(T)
    # ------------------------------------------------------------------
    def _compute_thermal_penalty(self, temperature: float) -> float:
        """Compute thermal penalty f_T(T) given T_high, T_red, alpha, beta."""
        # Comfort / cold: no thermal penalty
        if temperature <= self.T_high:
            return 0.0

        # Warning zone: T_high < T < T_red
        if temperature < self.T_red:
            arg = self.alpha * (temperature - self.T_high)
            arg = min(arg, self.max_exponent)
            return float(np.exp(arg) - 1.0)

        # Red zone: T >= T_red
        arg = self.beta * (temperature - self.T_red)
        arg = min(arg, self.max_exponent)
        return float(self.C_AL_calc + (np.exp(arg) - 1.0))

    # ------------------------------------------------------------------
    # Reward entry point
    # ------------------------------------------------------------------
    def __call__(self, obs_dict: dict):
        """
        Compute reward and associated info dictionary.

        Returns
        -------
        reward : float
            Reward value for the current timestep.
        info : dict[str, float]
            Logging dictionary containing:
                - 'reward'
                - 'comfort_penalty' / 'comfort_term'
                - 'energy_penalty' / 'energy_term'
                - 'total_temperature_violation'
                - 'total_power_demand'
                - 'max_temp'
                - 'power'
                - 'zone_status' (0=comfort, 1=warning, 2=red)
        """

        # --- Energy ---
        if self.energy_name not in obs_dict:
            raise KeyError(
                f"Energy key '{self.energy_name}' not found. "
                f"Available keys: {list(obs_dict.keys())}"
            )
        energy = float(obs_dict[self.energy_name])

        # --- Temperatures (multi-zone) ---
        temps = []
        for name in self.temp_names:
            if name not in obs_dict:
                raise KeyError(
                    f"Temperature key '{name}' not found. "
                    f"Available keys: {list(obs_dict.keys())}"
                )
            temps.append(float(obs_dict[name]))

        max_temp = max(temps)

        # --- Base penalties (costs) ---
        if self.energy_scale > 0:
            energy_penalty = energy / self.energy_scale
        else:
            energy_penalty = energy
        energy_penalty = float(energy_penalty)

        thermal_penalty = self._compute_thermal_penalty(max_temp)
        comfort_penalty = float(thermal_penalty)

        # Total reward (negative because we penalize costs)
        reward = -self.w_E * energy_penalty - self.w_T * comfort_penalty

        # Encoded comfort zone state (useful for logging)
        if max_temp <= self.T_high:
            zone_status = 0.0  # comfort
        elif max_temp < self.T_red:
            zone_status = 1.0  # warning
        else:
            zone_status = 2.0  # red

        # Info dict consumed by Sinergym / WandB (floats only)
        info = {
            "reward": float(reward),
            "comfort_penalty": comfort_penalty,
            "comfort_term": comfort_penalty,
            "energy_penalty": energy_penalty,
            "energy_term": energy_penalty,
            "total_temperature_violation": comfort_penalty,
            "total_power_demand": float(energy),
            "max_temp": float(max_temp),
            "power": float(energy),
            "zone_status": float(zone_status),
        }

        # Extra debug info (can include non-float values for manual inspection)
        self.last_info = {
            "reward": reward,
            "comfort_penalty": comfort_penalty,
            "energy_penalty": energy_penalty,
            "temps": temps,
            "max_temp": max_temp,
            "power": energy,
            "zone_status_str": (
                "comfort"
                if zone_status == 0.0
                else ("warning" if zone_status == 1.0 else "red")
            ),
        }

        return float(reward), info
