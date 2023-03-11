import visualize
import os
import neat
import pickle


local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-ctrnn')


with open('winner-ctrnn', 'rb') as f:
    winner = pickle.load(f)

    
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_path)
visualize.draw_net(config, winner, True, show_disabled = False, prune_unused = False)