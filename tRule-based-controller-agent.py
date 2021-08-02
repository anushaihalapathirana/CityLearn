from citylearn import  CityLearn
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from agents.rbc import RBC
import json
import time

""" RULE-BASED CONTROLLER (RBC) (Stores energy at night and releases it during the day)
 In this example, each building has its own RBC, which tries to flatten a generic building load 
 by storing energy at night and using it during the day, which isn't necessarily the best solution 
 in order to flatten the total load of the district."""


climate_zone = 5
sim_period = (0, 8760*4-1)
params = {'data_path':Path("data/Climate_Zone_"+str(climate_zone)), 
        'building_attributes':'building_attributes.json', 
        'weather_file':'weather_data.csv', 
        'solar_profile':'solar_generation_1kW.csv', 
        'carbon_intensity':'carbon_intensity.csv',
        'building_ids':["Building_"+str(i) for i in [1]],
        'buildings_states_actions':'buildings_state_action_space.json', 
        'simulation_period': sim_period, 
        'cost_function': ['ramping','1-load_factor','average_daily_peak','peak_demand','net_electricity_consumption','carbon_emissions'], 
        'central_agent': False,
        'save_memory': False,
        'verbose': 1}

env = CityLearn(**params)

observations_spaces, actions_spaces = env.get_state_action_spaces()

agents = RBC(actions_spaces)

# Finding which state 
with open('buildings_state_action_space.json') as file:
    actions_ = json.load(file)

indx_hour = -1
for obs_name, selected in list(actions_.values())[0]['states'].items():
    indx_hour += 1
    if obs_name=='hour':
        break
    assert indx_hour < len(list(actions_.values())[0]['states'].items()) - 1, "Please, select hour as a state for Building_1 to run the RBC"
        

state = env.reset()
done = False
rewards_list = []
start = time.time()
while not done:
    hour_state = np.array([[state[0][indx_hour]]])
    action = agents.select_action(hour_state)
    next_state, rewards, done, _ = env.step(action)
    state = next_state
    rewards_list.append(rewards)
cost_rbc = env.cost()
end = time.time()
print('cost rbc: ', cost_rbc)
print("run time: ", end-start)