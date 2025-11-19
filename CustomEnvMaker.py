import gymnasium as gym
import numpy as np
import os
from sinergym.utils.rewards import EnergyCostLinearReward

whtr = 'ITA_Rome.162420_IWEC.epw'

weather_path = os.path.join(os.getcwd(), 'weather_files', whtr)

new_action_space = gym.spaces.Box(
    low=np.array([15.0], dtype=np.float32),
    high=np.array([40.0], dtype=np.float32),
    shape=(1,),
    dtype=np.float32)

reward_parameters = {
    'temperature_variables': ['air_temperature'],
    'energy_variables': ['HVAC_electricity_demand_rate'],
    'energy_cost_variables': ['HVAC_electricity_demand_rate'], # 👈 ARGOMENTO MANCANTE
    'range_comfort_winter': [18, 27], 
    'range_comfort_summer': [18, 27], 
    'summer_start': [6, 1], 
    'summer_final': [9, 30], 
    'energy_weight': 0.3, 
    'lambda_energy': 1e-3, 
    'lambda_temperature': 1.0
}

env = gym.make('Eplus-smalldatacenter-aurora-continuous-v1',
               reward=EnergyCostLinearReward,
               env_name='AVIS_env',
               action_space=new_action_space,
               weather_files=[weather_path],
               reward_kwargs=reward_parameters
               )

print('New environment observation variables (time variables + variables + meters): {}'.format(
    env.get_wrapper_attr('observation_variables')))
print('New environment action variables (actuators): {}'.format(
    env.get_wrapper_attr('action_variables')))
for i in range(1):
    obs, info = env.reset()
    rewards = []
    truncated = terminated = False
    current_month = 0
    while not (terminated or truncated):
        a = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(a)
        rewards.append(reward)
    print(
        'Episode ',
        i,
        'Mean reward: ',
        np.mean(rewards),
        'Cumulative reward: ',
        sum(rewards))
env.close()
