from citylearn import  CityLearn
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from agents.rbc import RBC
import json

climate_zone = 5
sim_period = (0, 8759) # hourly time period to be simulated - 1 years 
params = {'data_path':Path("data/Climate_Zone_"+str(climate_zone)), 
        'building_attributes':'building_attributes.json', 
        'weather_file':'weather_data.csv', 
        'solar_profile':'solar_generation_1kW.csv', 
        'carbon_intensity':'carbon_intensity.csv',
        'building_ids':["Building_"+str(i) for i in [1,2,3,4,5,6,7,8,9]],
        'buildings_states_actions':'buildings_state_action_space.json',  # file containing the states and actions to be returned or taken by the environment
        'simulation_period': sim_period, 
        # ramping - sum of electricity consumption in every time step
        # 1-load_factor - average net electricity load / max electricity load
        # average_daily_peak - avg daily peak net demand
        'cost_function': ['ramping','1-load_factor','average_daily_peak','peak_demand','net_electricity_consumption','carbon_emissions','total'], # list of cost functions to be minimized
        'central_agent': False, # True returns list of states, single reward, and takes list of actions
        'save_memory': False,
        'verbose': 1 }

env = CityLearn(**params)

observations_spaces, actions_spaces = env.get_state_action_spaces()
print("observations_spaces:     ", observations_spaces)
print("actions_spaces:      ", actions_spaces)
print()

# Simulation without energy storage
env.reset()
done = False
while not done:
    obs, rewards, done, _ = env.step([[0 for _ in range(len(actions_spaces[i].sample()))] for i in range(9)])
costs = env.cost()


print("costs", costs)
print()
# print("building_information: ", env.get_building_information())
print()
print("baseline_cost: ", env.get_baseline_cost())
