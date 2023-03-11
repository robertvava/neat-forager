from audioop import avg
from classes.base import Environment
from classes.cells import Agent
import os
import neat
import multiprocessing
import pickle
import visualize
import numpy as np

ROWS = 42
COLS = 42
steps = 1000
runs_per_net = 5

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-ctrnn')


def eval_genome(genome, config):                                                    # config = config_path
    net = neat.ctrnn.CTRNN.create(genome, config, 1)                                # dt should be one.
    fitnesses = []
    # hps = [0]
    # losses = [0]
    for _ in range(runs_per_net):
        agent = Agent((int(ROWS//2), int(COLS//2)))
        env = Environment((ROWS, COLS), agent, max_steps = steps)
        net.reset()
        while env.current_step < env.max_steps:
            inputs =  list (agent.sensors_weights) + list(agent.factors) + list(agent.ext_factors) 
            update = net.advance(inputs, 1, time_step = 1) 
            env.step(agent, update)                                           # choice, steps = env.step
        fitness = ((np.mean(agent.food_consumed) - np.mean(agent.poison_consumed)) * agent.empty_steps/env.max_steps) + agent.hp/100 
        fitnesses.append(fitness)

    return max(fitnesses)

def eval_genomes(config):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_path)
    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(pe.evaluate)
    print('\nBest genome:\n{!s}'.format(winner))

    with open('winner-ctrnn', 'wb') as f:
        pickle.dump(winner, f)
    
    visualize.plot_stats(stats, ylog=True, view=True, filename="ctrnn-fitness.svg")
    visualize.plot_species(stats, view=True, filename="ctrnn-speciation.svg")

    # node_names = {-1: '-1', -2: '-2', -3: '-3', -4: '-4', 0: '0'} # Assign the names showed on the network graph. 
    visualize.draw_net(config, winner, True, show_disabled = False, prune_unused = False)

    return stats

if __name__ == '__main__':
    stats = eval_genomes(config_path)
    with open('stats', 'wb') as f:
        pickle.dump(stats, f)