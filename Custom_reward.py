import numpy as np
import torch
from typing import List, Union, Optional
from sinergym.utils.rewards import BaseReward


class ExponentialThermalReward(BaseReward):
    """
    Reward class used for **SAC training** in datacenter / HVAC control.

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
        # BaseReward (Sinergym v3+) does not receive env in the constructor
        super().__init__()

        # Weights for energy and thermal terms
        self.w_E = w_E
        self.w_T = w_T
        self.alpha = alpha
        self.beta = beta

        # One or multiple temperature variables
        if isinstance(temp_name, str):
            self.temp_names = [temp_name]
        else:
            self.temp_names = list(temp_name)

        # Energy variable and scaling factor
        self.energy_name = energy_name
        self.energy_scale = energy_scale

        # Thermal thresholds
        self.T_low = T_low
        self.T_high = T_high
        self.T_red = T_red

        # Numerical stability control for exponent arguments
        self.max_exponent = max_exponent

        # Ensure continuity between warning and red zone at T_red
        warning_at_red = min(
            self.alpha * (self.T_red - self.T_high),
            self.max_exponent,
        )
        # C_AL_calc is the value of the warning penalty at T_red
        self.C_AL_calc = float(np.exp(warning_at_red) - 1.0)

        # Internal debug info (not used directly by Sinergym / WandB)
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
            # Exponential growth within the warning range
            arg = self.alpha * (temperature - self.T_high)
            arg = min(arg, self.max_exponent)
            return float(np.exp(arg) - 1.0)

        # Red zone: T >= T_red
        # Stronger exponential penalty starting from T_red
        arg = self.beta * (temperature - self.T_red)
        arg = min(arg, self.max_exponent)
        return float(self.C_AL_calc + (np.exp(arg) - 1.0))

    # ------------------------------------------------------------------
    # Reward entry point
    # ------------------------------------------------------------------
    def __call__(self, obs_dict: dict):
        """
        Compute reward and associated info dictionary for one timestep.

        Parameters
        ----------
        obs_dict : dict
            Observation dictionary from Sinergym, containing energy and temperature keys.

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

        # --- Energy retrieval and validation ---
        if self.energy_name not in obs_dict:
            raise KeyError(
                f"Energy key '{self.energy_name}' not found. "
                f"Available keys: {list(obs_dict.keys())}"
            )
        energy = float(obs_dict[self.energy_name])

        # --- Temperatures (multi-zone support) ---
        temps = []
        for name in self.temp_names:
            if name not in obs_dict:
                raise KeyError(
                    f"Temperature key '{name}' not found. "
                    f"Available keys: {list(obs_dict.keys())}"
                )
            temps.append(float(obs_dict[name]))

        # Worst-case zone temperature
        max_temp = max(temps)

        # --- Base penalties (costs) ---
        if self.energy_scale > 0:
            energy_penalty = energy / self.energy_scale
        else:
            # Fallback: no scaling if energy_scale <= 0
            energy_penalty = energy
        energy_penalty = float(energy_penalty)

        # Thermal penalty from piecewise function
        thermal_penalty = self._compute_thermal_penalty(max_temp)
        comfort_penalty = float(thermal_penalty)

        # Total reward (negative sign because we penalize costs)
        reward = -self.w_E * energy_penalty - self.w_T * comfort_penalty

        # Encoded comfort zone state (useful for logging/analysis)
        if max_temp <= self.T_high:
            zone_status = 0.0  # comfort
        elif max_temp < self.T_red:
            zone_status = 1.0  # warning
        else:
            zone_status = 2.0  # red

        # Info dict consumed by Sinergym / WandB (must be float values)
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


class ESThermalEnergyReward(BaseReward):
    """
    Episodic thermal + energy fitness for **Evolution Strategies (ES)**, Sinergym-compatible.
    
    Version with graduated penalties and tight thresholds:
    - Comfort: [T_min_comfort, T_max_comfort] = [18.0, 26.5] °C
    - Zone 1 (soft):       26.5 - 27.5 °C
    - Zone 2 (moderate):   27.5 - 28.0 °C
    - Zone 3 (critical):   > 28.0 °C

    ENERGY FINE-TUNE PHASE:
    - If C >= C_min: F_temp = 0  → fitness is driven only by energy
    - If C < C_min: F_temp includes severity, violation and graduated penalties
    """

    def __init__(
        self,
        # Names of temperature variables (now MUST be a list with 2 zones in your current use case)
        temp_name: Union[str, List[str]],
        energy_name: str,
        baseline_energy_name: Optional[str] = None,
        baseline_energy_reference: Optional[float] = None,
        # Thermal thresholds (°C) - TIGHT VERSION
        T_zone1_low: float = 26.5,
        T_zone1_high: float = 27.5,
        T_zone2_high: float = 28.0,
        T_min_comfort: float = 18.0,
        T_max_comfort: float = 26.5,
        # Zone weights (w1 << w2 << w3)
        w1: float = 1.0,
        w2: float = 3.0,
        w3: float = 9.0,
        # Thermal fitness parameters
        alpha_zone: float = 1.0,
        lambda_peak: float = 0.0,  # effectively disabled in energy-focused phase
        beta_peak: float = 0.0,    # same as above
        # Comfort constraint
        C_min: float = 0.93,
        large_negative: float = -1e3,
        # GRADUATED PENALTIES for critical temperatures
        T_warning: float = 27.5,     # end of zone 1
        T_danger: float = 28.0,      # start of zone 3
        T_critical: float = 30.0,    # critical threshold
        penalty_mode: str = "exponential",  # currently not used in compute_episode_fitness
        # Combined fitness weights
        gamma_T: float = 1.0,
        gamma_E: float = 3.0,
        # Energy scale for step-wise reward (used when integrating with SB3/Sinergym, not ES)
        energy_scale_step: float = 10_000.0,
        # Debug mode toggle
        debug: bool = False,
    ):
        super().__init__()

        # Temperature variables: one or more zones
        if isinstance(temp_name, str):
            self.temp_names = [temp_name]
        else:
            self.temp_names = list(temp_name)

        # Energy variables and baselines for energy-saving computation
        self.energy_name = energy_name
        self.baseline_energy_name = baseline_energy_name
        self.baseline_energy_reference = baseline_energy_reference

        # Warn if energy fitness is active but no baseline is provided
        if gamma_E > 0:
            if baseline_energy_reference is None and baseline_energy_name is None:
                print(
                    "WARNING: gamma_E > 0 but no baseline provided. "
                    "Set baseline_energy_reference or baseline_energy_name. "
                    "Energy fitness will be disabled (F_energy=0)."
                )

        # Thermal zone thresholds and comfort band
        self.T_zone1_low = T_zone1_low
        self.T_zone1_high = T_zone1_high
        self.T_zone2_high = T_zone2_high
        self.T_min_comfort = T_min_comfort
        self.T_max_comfort = T_max_comfort

        # Zone weights for severity measure S_zone
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

        # Additional thermal shaping parameters (currently not fully used)
        self.alpha_zone = alpha_zone
        self.lambda_peak = lambda_peak
        self.beta_peak = beta_peak

        # Comfort constraint and large negative fallback
        self.C_min = C_min
        self.large_negative = large_negative

        # Graduated penalty thresholds
        self.T_warning = T_warning
        self.T_danger = T_danger
        self.T_critical = T_critical
        self.penalty_mode = penalty_mode

        # Final fitness combination weights
        self.gamma_T = gamma_T
        self.gamma_E = gamma_E

        # Energy scaling for step-wise reward
        self.energy_scale_step = energy_scale_step

        # Debug flag
        self.debug = debug

        # Episodic buffers for ES fitness computation
        self.reset_episode_buffers()

        # Last computed episodic metrics (for logging/inspection)
        self.last_episode_metrics = {}

    def reset_episode_buffers(self):
        """Reset per-episode buffers for a new ES episode."""
        self._temps = []
        self._energy = []
        self._energy_baseline = []

    def __call__(self, obs_dict: dict):
        """
        Per-timestep reward (used by Sinergym/SB3 wrappers).
        
        Returns a step-wise energy-focused reward, plus an info dict
        compatible with LoggerWrapper (includes 'comfort_term').
        """
        # --- Current energy ---
        if self.energy_name not in obs_dict:
            raise KeyError(f"Energy key '{self.energy_name}' not found.")
        energy = float(obs_dict[self.energy_name])

        # --- Temperatures (multi-zone, we track the max) ---
        temps = []
        for name in self.temp_names:
            if name not in obs_dict:
                raise KeyError(f"Temperature key '{name}' not found.")
            temps.append(float(obs_dict[name]))
        max_temp = max(temps)

        # --- Dynamic baseline energy (if provided per-timestep) ---
        baseline_energy = None
        if self.baseline_energy_name is not None:
            if self.baseline_energy_name in obs_dict:
                baseline_energy = float(obs_dict[self.baseline_energy_name])

        # --- Accumulate into episodic buffers (used for ES fitness) ---
        self._temps.append(max_temp)
        self._energy.append(energy)
        if baseline_energy is not None:
            self._energy_baseline.append(baseline_energy)

        # --- Step-wise reward: simple energy cost (for SB3 integration) ---
        if self.energy_scale_step > 0:
            energy_term = energy / self.energy_scale_step
        else:
            energy_term = energy

        step_reward = -float(energy_term)

        # --- Comfort term for LoggerWrapper compatibility ---
        # 0 if in comfort band, >0 if outside
        in_comfort = (max_temp >= self.T_min_comfort) and (max_temp <= self.T_max_comfort)
        if in_comfort:
            comfort_term = 0.0
        else:
            # Violation magnitude: distance from comfort band
            if max_temp < self.T_min_comfort:
                comfort_term = abs(self.T_min_comfort - max_temp)
            else:
                comfort_term = abs(max_temp - self.T_max_comfort)

        comfort_penalty = float(comfort_term)  # >0 if outside comfort, 0 if in comfort
        energy_penalty = float(energy_term)
        zone_status = 1.0 if not in_comfort else 0.0

        # Info dictionary consumed by Sinergym / LoggerWrapper / WandB
        info = {
            # raw per-step reward
            "reward": float(step_reward),
            "step_reward": float(step_reward),

            # comfort
            "comfort_penalty": comfort_penalty,
            "comfort_term": comfort_penalty,
            "total_temperature_violation": comfort_penalty,

            # energy
            "energy_penalty": energy_penalty,
            "energy_term": energy_penalty,
            "total_power_demand": float(energy),

            # instantaneous temperature and power
            "max_temp": float(max_temp),
            "step_max_temp": float(max_temp),
            "power": float(energy),
            "step_power": float(energy),

            # in/out comfort flag
            "zone_status": float(zone_status),
        }

        return float(step_reward), info

    def _compute_zone_indices(self, temps: np.ndarray) -> np.ndarray:
        """
        Assign, for each timestep, a zone index:
            0 : comfort  [T_min_comfort, T_max_comfort]
            1 : 26.5 < T <= 27.5
            2 : 27.5 < T <= 28.0
            3 : T > 28.0
           -1 : below comfort (T < T_min_comfort)
        """
        T = temps
        zone = np.zeros_like(T, dtype=int)
        zone[:] = 0  # default: comfort

        # Above comfort
        zone[(T > self.T_zone1_low) & (T <= self.T_zone1_high)] = 1
        zone[(T > self.T_zone1_high) & (T <= self.T_zone2_high)] = 2
        zone[T > self.T_zone2_high] = 3

        # Below comfort
        zone[T < self.T_min_comfort] = -1
        return zone

    def _compute_graduated_penalty(self, Tmax: float) -> float:
        """
        Compute a continuous graduated penalty to avoid cliffs in the fitness.
        Used only when C < C_min (in stricter modes).
        Currently not used in the ENERGY-SOFT version of compute_episode_fitness,
        but kept for compatibility and future extensions.
        """
        if Tmax <= self.T_max_comfort:
            return 0.0
        
        # 1. SOFT ZONE (26.5 - 27.5): light linear penalty
        if Tmax <= self.T_warning:
            # Max penalty here: 1.0 * 1.0 = 1.0
            return 1.0 * (Tmax - self.T_max_comfort)
            
        # 2. WARNING ZONE (27.5 - 28.0): moderate linear penalty
        elif Tmax <= self.T_danger:
            # Base carried from previous zone
            base = 1.0 * (self.T_warning - self.T_max_comfort)  # = 1.0
            # Extra: slope 10
            # Max here: 1.0 + 10.0 * 0.5 = 6.0
            return base + 10.0 * (Tmax - self.T_warning)
            
        # 3. DANGER ZONE (> 28.0): quadratic penalty (smooth but strong)
        else:
            # Base carried up to T_danger
            base = 1.0 * (self.T_warning - self.T_max_comfort) + \
                   10.0 * (self.T_danger - self.T_warning)  # = 1.0 + 5.0 = 6.0
            
            # Quadratic penalty for strong punishment of large outliers
            return base + 5.0 * ((Tmax - self.T_danger) ** 2)

    def compute_episode_fitness(self):
        """
        ENERGY-DRIVEN SOFT MODE:
        
        - The episode is evaluated primarily via F_energy (energy saving).
        - Comfort appears only as a soft penalty:
            violation = 1 - C          (fraction of time outside comfort)
            S_zone                     (weighted time in zones 1–3)
            Tmax > T_max_comfort       (small penalty on peaks)
        
        F_temp = -(violation) - 0.1 * S_zone - 0.05 * max(0, Tmax - T_max_comfort)
        F_tot  = gamma_E * F_energy + gamma_T * F_temp
        
        No hard constraint on C_min: C_min is kept only for logging/diagnostics.
        """
        if len(self._temps) == 0:
            # Safety fallback if episode buffers are empty
            print("⚠️  WARNING: Empty episode buffer!")
            self.last_episode_metrics = {
                "C": 0.0, "f1": 0.0, "f2": 0.0, "f3": 0.0, "S_zone": 0.0,
                "Tmax": None, "P_peak": 0.0, "P_graduated": 0.0, "Esaving": None,
                "F_temp": None, "F_energy": None, "F_total": self.large_negative,
                "comfort_constraint_violated": False,
                "baseline_mode": "reference" if self.baseline_energy_reference else "dynamic",
            }
            return float(self.large_negative), self.last_episode_metrics

        temps = np.array(self._temps, dtype=float)
        energy = np.array(self._energy, dtype=float)

        # Zone indices and fractions of time in each zone
        zone_idx = self._compute_zone_indices(temps)
        f1 = float(np.mean(zone_idx == 1))
        f2 = float(np.mean(zone_idx == 2))
        f3 = float(np.mean(zone_idx == 3))

        # Comfort fraction C
        in_comfort = (temps >= self.T_min_comfort) & (temps <= self.T_max_comfort)
        C = float(np.mean(in_comfort))

        # Severity measure S_zone with zone weights
        S_zone = self.w1 * f1 + self.w2 * f2 + self.w3 * f3

        # Maximum episode temperature
        Tmax = float(np.max(temps))

        if self.debug:
            print(f"\n[DEBUG] Temperature Statistics:")
            print(f"  Min: {np.min(temps):.2f}°C")
            print(f"  Mean: {np.mean(temps):.2f}°C")
            print(f"  Max: {Tmax:.2f}°C")
            print(f"  Std: {np.std(temps):.2f}°C")
            print(f"  Total timesteps: {len(temps)}")

        # --- Soft comfort penalty ---
        # violation = fraction of time outside comfort
        violation = 1.0 - C
        # Temperature overshoot during the worst peak
        over_T = max(0.0, Tmax - self.T_max_comfort)

        # P_graduated is kept for compatibility; here it is "soft"
        P_graduated = 0.1 * S_zone + 0.05 * over_T

        # Soft F_temp: no separate paths, no hard use of C_min in the formula
        F_temp_base = -violation - 0.1 * S_zone - 0.05 * over_T

        # Constraint gap used as a soft penalty term
        constraint_gap = max(0.0, self.C_min - C)
        k_constraint = 10.0

        F_temp = F_temp_base - k_constraint * constraint_gap

        comfort_constraint_violated = C < self.C_min

        # --- F_energy: energy-saving based episodic term ---
        Esaving = None
        F_energy = 0.0

        # Case 1: static baseline reference
        if self.baseline_energy_reference is not None:
            sum_agent = float(np.sum(energy))
            if self.baseline_energy_reference > 0:
                Esaving = 1.0 - (sum_agent / self.baseline_energy_reference)
                F_energy = Esaving

        # Case 2: dynamic baseline from environment
        elif len(self._energy_baseline) == len(temps):
            base = np.array(self._energy_baseline, dtype=float)
            sum_base = float(np.sum(base))
            sum_agent = float(np.sum(energy))
            if sum_base > 0:
                Esaving = 1.0 - (sum_agent / sum_base)
                F_energy = Esaving

        # --- Total episodic fitness ---
        F_total = self.gamma_T * F_temp + self.gamma_E * F_energy

        if self.debug:
            print(f"\n{'='*60}")
            print(f"[REWARD DEBUG] Episode Summary (ENERGY-SOFT MODE):")
            print(f"  C (comfort rate): {C:.4f} (C_min: {self.C_min})")
            print(f"  violation (1-C) : {violation:.4f}")
            print(f"  S_zone          : {S_zone:.4f}")
            print(f"  Tmax            : {Tmax:.2f}°C (comfort limit: {self.T_max_comfort}°C)")
            print(f"  over_T          : {over_T:.2f}")
            print(f"  P_graduated     : {P_graduated:.4f}")
            print(f"  F_temp          : {F_temp:.4f}")
            print(f"  F_energy        : {F_energy:.4f}")
            print(f"  F_total         : {F_total:.4f}")
            print(f"{'='*60}\n")

        metrics = {
            "C": C, "f1": f1, "f2": f2, "f3": f3, "S_zone": S_zone,
            "Tmax": Tmax, "P_peak": 0.0, "P_graduated": P_graduated,
            "Esaving": Esaving,
            "F_temp": F_temp, "F_energy": F_energy, "F_total": F_total,
            "comfort_constraint_violated": comfort_constraint_violated,
            "baseline_mode": "reference" if self.baseline_energy_reference else "dynamic",
        }

        self.last_episode_metrics = metrics
        return float(F_total), metrics


def rank_based_utilities(all_rewards: np.ndarray) -> torch.Tensor:
    """
    Rank-based utility function for Evolution Strategies.

    Parameters
    ----------
    all_rewards : np.ndarray
        Shape (N,), episodic rewards (possibly already clipped).

    Returns
    -------
    torch.Tensor
        Shape (N,), rank-based utilities with mean ~ 0 and variance ~ 1.
        Higher reward episodes get higher utility.
    """
    N = all_rewards.shape[0]

    # Rank 0 = best, N-1 = worst
    # argsort on -all_rewards so that highest reward has rank 0
    ranks = np.argsort(np.argsort(-all_rewards))

    # Linear utilities: best → high value, worst → low value
    # u_i = (N - 1 - rank_i) - (N - 1)/2  ⇒ zero mean
    utilities = (N - 1 - ranks) - (N - 1) / 2.0

    # Normalize to variance ~ 1 (optional but usually helpful)
    utilities = utilities / (utilities.std() + 1e-8)

    return torch.tensor(utilities, dtype=torch.float32)
