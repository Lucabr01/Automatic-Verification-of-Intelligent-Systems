import os
import sys
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, BaseCallback
from stable_baselines3.common.logger import HumanOutputFormat
from stable_baselines3.common.logger import Logger as SB3Logger

import sinergym
from Custom_reward import ExponentialThermalReward

from sinergym.utils.wrappers import (
    NormalizeObservation,
    NormalizeAction,
    LoggerWrapper,
    CSVLogger,
    WandBLogger,
    is_wrapped,
)
from sinergym.utils.callbacks import LoggerEvalCallback
from sinergym.utils.logger import WandBOutputFormat


# ============================================================================
# CURRICULUM LEARNING CALLBACK (2 PHASES)
# ============================================================================
class CurriculumLearningCallback(BaseCallback):
    """
    Two-phase curriculum learning on the reward function:

    Phase 1:
        - Comfort and energy are penalized with similar weights.
    Phase 2:
        - Energy is penalized slightly more than comfort.

    The callback updates the reward parameters in both training and evaluation
    environments once the transition step is reached.
    """

    def __init__(self, env, eval_env, phase_transition: int = 200_000, verbose: int = 1):
        super().__init__(verbose)
        self.env = env
        self.eval_env = eval_env
        self.phase_transition = phase_transition
        self.current_phase = 1

        self.phases = {
            1: {
                "name": "PHASE 1: Balanced Comfort & Energy",
                "w_E": 1.0,
                "w_T": 1.0,
                "energy_scale": 17_500.0,
                "description": "Learn basic control with similar penalties for comfort and energy.",
            },
            2: {
                "name": "PHASE 2: Mild Energy Priority",
                "w_E": 1.5,
                "w_T": 1.0,
                "energy_scale": 17_000.0,
                "description": "Energy is slightly more penalized than comfort.",
            },
        }

    def _on_step(self) -> bool:
        # Transition Phase 1 -> Phase 2
        if self.num_timesteps >= self.phase_transition and self.current_phase == 1:
            self._transition_to_phase(2)

        # Periodic logging
        if self.num_timesteps % 10_000 == 0:
            phase = self.phases[self.current_phase]
            print("\n" + "=" * 70)
            print(f"Step {self.num_timesteps:,} | {phase['name']}")
            print(f"w_E={phase['w_E']} | w_T={phase['w_T']} | scale={phase['energy_scale']}")
            print("=" * 70)
        return True

    def _transition_to_phase(self, new_phase: int) -> None:
        self.current_phase = new_phase
        p = self.phases[new_phase]

        print("\n" + "=" * 70)
        print(f"CURRICULUM TRANSITION -> {p['name']}")
        print(f"Step: {self.num_timesteps:,}")
        print(f"Description: {p['description']}")
        print(f"New reward weights: w_E={p['w_E']} | w_T={p['w_T']} | scale={p['energy_scale']}")
        print("=" * 70 + "\n")

        self._update_env_reward_params(self.env, p)
        self._update_env_reward_params(self.eval_env, p)

        # Log phase change to Weights & Biases if available
        if is_wrapped(self.env, WandBLogger):
            run = self.env.get_wrapper_attr("wandb_run")
            run.log(
                {
                    "phase": new_phase,
                    "w_E": p["w_E"],
                    "w_T": p["w_T"],
                    "energy_scale": p["energy_scale"],
                },
                step=self.num_timesteps,
            )

    def _update_env_reward_params(self, env, p: dict) -> None:
        """Update reward parameters inside the environment's reward function."""
        try:
            reward_fn = env.get_wrapper_attr("reward_fn")
            if reward_fn is not None:
                reward_fn.W_energy = p["w_E"]
                reward_fn.W_comfort = p["w_T"]
                reward_fn.energy_scale = p["energy_scale"]
        except Exception as e:
            print("Warning: error while updating reward parameters:", e)


# ============================================================================
# DEVICE CONFIG
# ============================================================================
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    print("WARNING: CUDA not available, using CPU.")
    device = "cpu"


# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================
ENV_ID = "Eplus-datacenter_dx-mixed-continuous-v1"

# Initial reward parameters (Phase 1)
reward_parameters = {
    "w_E": 1.0,
    "w_T": 1.0,
    "energy_scale": 17_500.0,
    "T_high": 27.0,
    "T_red": 28.0,
    "temp_name": ["east_zone_air_temperature", "west_zone_air_temperature"],
    "energy_name": "HVAC_electricity_demand_rate",
}

episodes = 80
experiment_date = datetime.today().strftime("%Y-%m-%d_%H%M")
experiment_name = f"CURRICULUM_SAC_{experiment_date}"

# Actuators controlled by the RL agent
new_actuators = {
    "Cooling_Setpoint_RL": (
        "Schedule:Compact",
        "Schedule Value",
        "Cooling Setpoints",
    ),
    "East_Zone_Fan_Flow": (
        "Fan",
        "Fan Air Mass Flow Rate",
        "EAST ZONE SUPPLY FAN",
    ),
    "West_Zone_Fan_Flow": (
        "Fan",
        "Fan Air Mass Flow Rate",
        "WEST ZONE SUPPLY FAN",
    ),
}

# Continuous action space: [cooling setpoint, east fan flow, west fan flow]
new_action_space = gym.spaces.Box(
    low=np.array([20.0, 1.0, 1.0], dtype=np.float32),
    high=np.array([30.0, 7.0, 7.0], dtype=np.float32),
    dtype=np.float32,
)

env_kwargs = dict(
    reward=ExponentialThermalReward,
    reward_kwargs=reward_parameters,
    actuators=new_actuators,
    action_space=new_action_space,
)

print("\n" + "=" * 70)
print("CURRICULUM TRAINING CONFIG")
print(f"Environment: {ENV_ID}")
print("Phases:")
print(" 1) Balanced:      w_E=1.0,  w_T=1.0")
print(" 2) Mild energy:   w_E=1.5,  w_T=1.0")
print("=" * 70 + "\n")

env = gym.make(ENV_ID, env_name=experiment_name, **env_kwargs)
eval_env = gym.make(
    ENV_ID,
    env_name=experiment_name + "_EVAL",
    **env_kwargs,
)

print("Environments created.")


# ============================================================================
# WRAPPERS
# ============================================================================
print("\nApplying wrappers...")

env = NormalizeObservation(env)
env = NormalizeAction(env)
env = LoggerWrapper(env)
env = CSVLogger(env)
env = Monitor(env)

USE_WANDB = True
if USE_WANDB:
    try:
        env = WandBLogger(
            env,
            entity="lucabrunetti2001-n",
            project_name="AVIS",
            run_name=experiment_name,
            group="DatacenterDX_Curriculum",
            tags=["SAC", "curriculum_learning", "datacenter_dx"],
            save_code=True,
        )
        print("Weights & Biases logging enabled.")
    except Exception as e:
        print("Warning: could not initialize Weights & Biases logging:", e)
        USE_WANDB = False

eval_env = NormalizeObservation(eval_env)
eval_env = NormalizeAction(eval_env)
eval_env = LoggerWrapper(eval_env)
eval_env = CSVLogger(eval_env)
eval_env = Monitor(eval_env)

print("Wrappers applied.\n")


# ============================================================================
# SAC MODEL
# ============================================================================
policy_kwargs = dict(
    activation_fn=torch.nn.Tanh,
    net_arch=dict(pi=[1024, 512], qf=[1024, 512]),
)

model = SAC(
    "MlpPolicy",
    env=env,
    learning_rate=1e-4,
    buffer_size=1_000_000,
    learning_starts=10_000,
    batch_size=512,
    tau=0.002,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    ent_coef="auto",
    policy_kwargs=policy_kwargs,
    verbose=1,
    device=device,
    tensorboard_log=f"./tensorboard_logs/{experiment_name}",
)

print("SAC model created.")


# ============================================================================
# CALLBACKS
# ============================================================================
curriculum_callback = CurriculumLearningCallback(
    env=env,
    eval_env=eval_env,
    phase_transition=110_000,
    verbose=1,
)

eval_callback = LoggerEvalCallback(
    eval_env=eval_env,
    train_env=env,
    n_eval_episodes=2,
    eval_freq_episodes=5,
    deterministic=True,
)

callback = CallbackList([curriculum_callback, eval_callback])

if is_wrapped(env, WandBLogger):
    logger = SB3Logger(
        folder=None,
        output_formats=[
            HumanOutputFormat(sys.stdout, max_length=120),
            WandBOutputFormat(),
        ],
    )
    model.set_logger(logger)


# ============================================================================
# TRAINING
# ============================================================================
steps_per_episode = env.get_wrapper_attr("timestep_per_episode")
total_timesteps = min(episodes * steps_per_episode, 500_000)

print("\n" + "=" * 70)
print("STARTING CURRICULUM TRAINING")
print(f"Total timesteps: {total_timesteps:,}")
print("=" * 70 + "\n")

try:
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        log_interval=1,
        progress_bar=True,
    )

    workspace_path = env.get_wrapper_attr("workspace_path")
    save_path = os.path.join(workspace_path, "model_curriculum2_sac")
    model.save(save_path)
    print(f"\nModel saved at: {save_path}")

except Exception as e:
    print("Training error:", e)
finally:
    env.close()
    eval_env.close()
    print("Training finished.")
