import os
import matplotlib.pylab as plt
from gui.ui import run_ui
from classes.base import Environment 
from classes.cells import Agent
import neat
import numpy as np
import pickle 
from mpl_toolkits.axes_grid1 import make_axes_locatable 


with open('winner-ctrnn', 'rb') as f:
    g = pickle.load(f)

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-ctrnn')

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)
net = neat.ctrnn.CTRNN.create(g, config, 1)

ROWS = 42
COLS = 42
steps = 1000
agent = Agent((int(ROWS//2), int(COLS//2)))
env = Environment((ROWS, COLS), agent, max_steps = steps)

run_ui(net, 1000, agent, env, 0.00001, ROWS, COLS)

fitness = ((np.mean(agent.food_consumed) - np.mean(agent.poison_consumed)) * agent.empty_steps/env.max_steps) + agent.hp/100 
print (fitness)



food_weights_list = [agent.sensors_weights[0]]
poison_weights_list = [agent.sensors_weights[1]]
food_factor_weight_list = [0.5]
poison_factor_weight_list = [0.5]
empty_factor_weight_list = [0.5]
food_ext_factor_weight_list = [0.5]
poison_ext_factor_weight_list = [0.5]
empty_ext_factor_weight_list = [0.5]
# updates_list = []

# Lists from the CTRNN
food_factor_weight_update_list = [0]
poison_factor_weight_update_list = [0]
empty_factor_weight_update_list = [0]
food_ext_factor_weight_update_list = [0]
poison_ext_factor_weight_update_list = [0]
empty_ext_factor_weight_update_list = [0]
path = [agent.position]
choice_weights = [1]

while env.current_step < env.max_steps:
    inputs = list(agent.sensors_weights) + list(agent.factors) + list(agent.ext_factors)
    update = net.advance(inputs, 1, 1)
    choose = env.step(agent, update)
    path.append(agent.position)
    
    poison_weights_list.append(agent.sensors_weights[0])
    food_weights_list.append(agent.sensors_weights[1])
    food_factor_weight_update_list.append(update[2])
    poison_factor_weight_update_list.append(update[3])
    empty_factor_weight_update_list.append(update[4])
    food_ext_factor_weight_update_list.append(update[5])
    poison_ext_factor_weight_update_list.append(update[6])
    empty_ext_factor_weight_update_list.append(update[7])
    food_factor_weight_list.append(agent.factors[0])
    poison_factor_weight_list.append(agent.factors[1])
    empty_factor_weight_list.append(agent.factors[2])
    food_ext_factor_weight_list.append(agent.ext_factors[0])
    poison_ext_factor_weight_list.append(agent.ext_factors[1])
    empty_ext_factor_weight_list.append(agent.ext_factors[2])
    choice_weights.append(choose.weight)
    # updates_list.append(update)


def plot_graphs():
    t = [i for i in range(env.current_step+1)]
    plt.figure(figsize = (15, 10))
    plt.plot(t[:50], list(food_weights_list)[:50], 'r', label = "F")
    plt.plot(t[:50], list(poison_weights_list)[:50], 'g', label = "P")
    plt.title("Food sensors weights")
    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Weight")

    plt.figure(figsize = (15, 10))
    plt.plot(t, food_factor_weight_update_list, 'b', label = 'Food factor update')
    plt.plot(t, poison_factor_weight_update_list, 'y', label = 'Poison factor update')
    plt.plot(t, empty_factor_weight_update_list, label = 'Empty factor update')
    plt.title("Update rule for neighbourhood")
    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Update value")

    plt.figure(figsize = (15, 10))
    plt.plot(t,food_ext_factor_weight_update_list, label = 'Food extended factor update')
    plt.plot(t,poison_ext_factor_weight_update_list, label = 'Poison extended factor update')
    plt.plot(t,empty_ext_factor_weight_update_list, label = 'Step extended factor update')
    plt.title("Update rule for extended neighbourhood")
    plt.xlabel("Steps")
    plt.ylabel("Extended factor value")
    plt.legend()

    plt.figure(figsize = (15, 10))
    plt.plot(t,food_ext_factor_weight_update_list, label = 'Ext food update')
    plt.plot(t, food_factor_weight_update_list, 'b', label = 'Food weight')
    plt.title("Food sensors weights")
    plt.xlabel("Steps")
    plt.ylabel("Weight")
    plt.legend()
    
    xs = []
    ys = []
    for i in range(len(path)):
        xs.append(path[i][0])
        ys.append(path[i][1])
    cm_subsection = np.linspace(0, env.current_step, env.current_step) 
    plt.figure(figsize = (15, 10))
    plt.scatter(xs, ys, s = 50, c=t, cmap='cividis', alpha = 0.5)  
    plt.show()

plot_graphs()
