import numpy as np
import random
from classes.cells import Cell, Consumable


# A class that sets up the environment in which the agent operates. It receives height and weight dimensions (for now, they should be the same), a max number of steps and the agent class instantiation. 
class Environment:
    def __init__(self, dim, agent, max_steps):
        self.dim = (dim[0]+1, dim[1]+1)
        self.cells = self.fill_cells()
        self.generate_cells(agent)
        self.current_step = 0
        self.max_steps = max_steps

    # Fill the space with the Cell class and with 0s on the matrix edges. 
    def fill_cells(self):
        cell_matrix = [[Cell((x, y)) for y in range(self.dim[0])] for x in range(self.dim[1])]
        for i in range(self.dim[0]):
            cell_matrix[0][i] = 0
            cell_matrix[1][i] = 0
            cell_matrix[i][self.dim[1]-1] = 0
            cell_matrix[self.dim[0]-1][i] = 0
            cell_matrix[self.dim[0]-2][i] = 0
            for j in range(self.dim[1]):
                cell_matrix[j][0] = 0
                cell_matrix[j][1] = 0
                cell_matrix[j][-1] = 0
                cell_matrix[j][-2] = 0
        return cell_matrix

    # Generate food and poison at random.       
    def generate_fp(self, agent, penalty = 1):
        # Retain agent's position to avoid deleting it from the matrix. 
        occupied = agent.position    
        # Generate random (row, col) positions for foods and poisons. 
        rows = np.random.randint(self.dim[0] - 1, size = int(self.dim[0]))  
        cols = np.random.randint(self.dim[1] - 1, size = int(self.dim[1]))
        # Generate random quantities within a certain range. The food/poison proportion can be adjusted with the penalty value (of type integer). 
        quantities = np.random.randint(5, size = (int((self.dim[0]+self.dim[1])/2)))
        types = ['P' for i in range((len(rows)+penalty))] + ['F' for j in range(len(cols)-penalty)]
        random.shuffle(types) 
        for i in range(len(rows)):  
            if (rows[i], cols[i]) == occupied or isinstance(self.cells[rows[i]][cols[i]], int):  # Avoid generating Consumables over the agent or on matrix edges. 
                pass
            else:
                self.cells[rows[i]][cols[i]] = Consumable((rows[i], cols[i]), types[i], quantities[i]) # Assign Consumable of random type, quantity and position. 

    # Generate the food, poison and copy-paste agent instantiation for avoiding position conflicts.
    def generate_cells(self, agent):
        self.generate_fp(agent)
        self.cells[agent.position[0]][agent.position[1]] = agent

    # Swap agent with empty cells. This allows the agent to effectively move around the space at each time step. 
    def swap(self, agent, cell):
        if isinstance(cell, int):
            pass
        mempos = cell.position
        agentpos = agent.position
        cell.position = agentpos
        agent.position = mempos
        self.cells[mempos[0]][mempos[1]] = agent
        self.cells[agentpos[0]][agentpos[1]] = cell

    # Allows the agent to interact with the chosen cell. 
    def interact(self, agent, cell):
        if isinstance(cell, int):      # Avoid interacting with non-cells. 
            pass
        if cell.state == 0:            # Allow the agent to swap with empty cells. 
            self.swap(agent, cell)
        elif cell.state == 1:          # Safety measure to avoid conflicts. 
            pass
        elif cell.state == 2:
            ate = agent.consume_cell(cell)
            cell.consumed(ate)      
            if cell.is_empty():
                cell.state = 0
                self.cells[cell.position[0]][cell.position[1]] = Cell((cell.position[0], cell.position[1]))
                cell = self.cells[cell.position[0]][cell.position[1]]
                self.swap(agent, cell)

    # Updates the weight map associated with the cell matrix. Effectively, this is equivalent to the agent's memory. 
    def update_env_weights(self, nbh, ext_nbh):
        for i in range(len(nbh[1])):                # Update neighbourhood weights
            for j in range(len(self.cells[0])):
                for k in range(len(self.cells[1])):
                    if not isinstance(self.cells[j][k], int):
                        if self.cells[j][k].position == nbh[1][i].position and self.cells[j][k].state != 1:
                            self.cells[j][k].weight = nbh[0][i]
        for i in range(len(ext_nbh[1])):            # Update extended neighbourhood weights
            for j in range(len(self.cells[0])):
                for k in range(len(self.cells[1])):
                    if not isinstance(self.cells[j][k], int):
                        if self.cells[j][k].position == ext_nbh[1][i].position and self.cells[j][k].state != 1:
                            self.cells[j][k].weight = ext_nbh[0][i]

    '''
    The agent step function contains multiple actions. 
    1. A penalty for the agent's movement - subtracting a set value from the agent's hp. 2
    2. Updating the agent's sensors based on the CTRNN outputs. 
    3. Get the neighbourhood and extended neighbourhood. 
    4. Inpsect the extended neighbourhood to verify if it is worth considering as a future choice. 
    5. The agent decides what step to take next by inspecting the neighbourhood and extended neighbourhood.
    6. Update weights based on the 24 (or 8) options considered. The number of options can change based on whether there are surrounding non-cells. 
    7. Update empty steps and current step for advancing the simulation
    8. At certain steps, generate more food and poison. 
    9. Return choice for updating the neural network. 
    '''

    def step(self, agent, act):
        agent.hp -= 0.05
        agent.update_sensors(act)
        nbh = agent.get_neighbourhood(self.cells)
        ext_nbh = agent.get_extended_neighbourhood(self.cells)
        inspect = agent.inspect(ext_nbh)
        choice, options, ext_options = agent.decide_step(self.cells, nbh, ext_nbh, inspect)
        self.update_env_weights(options, ext_options)
        self.interact(agent, choice)
        if choice.state == 0:
            agent.empty_steps += 1
        self.current_step += 1
        if self.current_step in [self.max_steps/4, self.max_steps/2, self.max_steps/8]:
            self.generate_fp(agent)
        return choice
    