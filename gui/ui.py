import sys
sys.path.insert(1, '..\ca_env')
import pygame
from classes.base import Environment 
from classes.cells import Agent
import time
import neat
import numpy as np
import pickle
import os
from gui.constants import BLACK, GREY, GREEN, RED, BLUE, MARGIN


# with open('winner-ctrnn', 'rb') as f:
#     g = pickle.load(f)

# local_dir = os.path.dirname(__file__)
# config_path = os.path.join(local_dir, 'config-ctrnn')

# config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
#                      neat.DefaultSpeciesSet, neat.DefaultStagnation,
#                      config_path)

# # steps = 1000
# # ROWS = 32
# # COLS = 32

# net = neat.ctrnn.CTRNN.create(g, config, 1)

def run_ui(net, steps, agent, env, dt, ROWS, COLS):
        
    pygame.init()
    # Set the HEIGHT and WIDTH of the screen
    WINDOW_SIZE = [1000, 1000]                                    # these values were set for a LARGE screen. 
    WIDTH = WINDOW_SIZE[0]//env.dim[0]-1.5
    HEIGHT = WINDOW_SIZE[1]//env.dim[1]-1.5

    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Game of Life Simulation")
    running = True
    clock = pygame.time.Clock() 

    while running and env.current_step < env.max_steps:

        for event in pygame.event.get():  
            if event.type == pygame.QUIT:  
                running = False  
        screen.fill(BLACK)
        # Draw the grid
        
        for row in range(env.dim[0]):
            for col in range(env.dim[1]):
                if type(env.cells[row][col]) != (int):
                    if env.cells[row][col].state == 0:
                        color = GREY
                    elif env.cells[row][col].state == 1:
                        color = BLUE
                    elif env.cells[row][col].state == 2:
                        if env.cells[row][col].type == 'F':
                            color = GREEN
                        elif env.cells[row][col].type == 'P':
                            color = RED
                else:
                    color = BLACK
                pygame.draw.rect(screen,
                                color,
                                [(MARGIN + WIDTH) * col + MARGIN,
                                (MARGIN + HEIGHT) * row + MARGIN,
                                WIDTH,
                                HEIGHT])
       
        inputs =  list (agent.sensors_weights) + list(agent.factors) + list(agent.ext_factors)
        update = net.advance(inputs, 1, time_step = 1) 
        env.step(agent, update)                                           # choice, steps = env.step
        fitness = (np.mean(agent.food_consumed) - np.mean(agent.poison_consumed)) * agent.empty_steps/env.max_steps  


        if agent.hp <= 0:
            running = False
        # Frames per second limit
        clock.tick(60)
        time.sleep(dt)
        # Update screen
        pygame.display.flip()

    # Exit, otherwise we'll all get stuck
    pygame.quit()
    return agent.sensors, fitness


# ROWS = 32
# COLS = 32
# steps = 1000
# agent = Agent((int(ROWS//2), int(COLS//2)))
# env = Environment((ROWS, COLS), agent, max_steps = steps)

# run_ui(net, steps, agent, env, 0.001, ROWS+2, COLS+2)