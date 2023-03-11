from classes.base import Environment
from classes.cells import Agent
import neat
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from gui.ui import run_ui

with open('winner-ctrnn', 'rb') as f:
    g = pickle.load(f)


local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-ctrnn')

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

steps = 1000
ROWS = 32
COLS = 32


agent = Agent((int(ROWS//2), int(COLS//2)))
env = Environment((ROWS, COLS), agent, max_steps = steps)

net = neat.ctrnn.CTRNN.create(g, config, 1)

weights_hist = []
sensors_hist = []
activations_hist = []
fitnesses = []
hps = []
while env.current_step < env.max_steps:
    inputs = list(agent.sensors) + list(agent.factors) + [agent.eating_weight]
    action = net.advance(inputs, 1, time_step = 1) 
    env.step(agent, action)                                           # choice, steps = env.step
    fitness = - np.mean(agent.poison_consumed) - agent.empty_steps/env.max_steps + np.mean(agent.food_consumed) 
    if env.current_step in [500, 1000, 250, 600, 100, 50, 10, 5, 2, 25]:
        print ("Facotrs: ")
        print (agent.factors)
        print ("Sensors at " + str(env.current_step))
        print (agent.sensors)                            




# run_ui(net, 1000, agent, env, 0.5, ROWS, COLS)






