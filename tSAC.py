from citylearn import  CityLearn
from pathlib import Path
import numpy as np                                                                                                                                                                                      
import torch
import matplotlib.pyplot as plt
from agents.sac import SAC as Agent

# Load environment
climate_zone = 5
params = {'data_path':Path("data/Climate_Zone_"+str(climate_zone)), 
        'building_attributes':'building_attributes.json', 
        'weather_file':'weather_data.csv', 
        'solar_profile':'solar_generation_1kW.csv', 
        'carbon_intensity':'carbon_intensity.csv',
        'building_ids':["Building_"+str(i) for i in [1,2]],
        'buildings_states_actions':'buildings_state_action_space.json', 
        'simulation_period': (0, 8759), 
        'cost_function': ['ramping','1-load_factor','average_daily_peak','peak_demand','net_electricity_consumption','carbon_emissions'], 
        'central_agent': False,
        'save_memory': False ,
        'verbose':1}

# Contain the lower and upper bounds of the states and actions, to be provided to the agent to normalize the variables between 0 and 1.
env = CityLearn(**params)
observations_spaces, actions_spaces = env.get_state_action_spaces()

# Provides information on Building type, Climate Zone, Annual DHW demand, Annual Cooling Demand, Annual Electricity Demand, Solar Capacity, and correllations among buildings
building_info = env.get_building_information()

params_agent = {'building_ids':["Building_"+str(i) for i in [1,2]],
                 'buildings_states_actions':'buildings_state_action_space.json', 
                 'building_info':building_info,
                 'observation_spaces':observations_spaces, 
                 'action_spaces':actions_spaces}

# Instantiating the control agent(s)
agents = Agent(**params_agent)

state = env.reset()
done = False

action, coordination_vars = agents.select_action(state)    
while not done:
    next_state, reward, done, _ = env.step(action)
    action_next, coordination_vars_next = agents.select_action(next_state)
    agents.add_to_buffer(state, action, reward, next_state, done, coordination_vars, coordination_vars_next)
    coordination_vars = coordination_vars_next
    state = next_state
    action = action_next

cost = env.cost()

print('cost: ', cost)