import numpy as np


"""
Cells can take the states: 
0 = Empty cell 
1 = Agent Cell filled with Agent class
2 = Food or Poison cell filled with Consumable class of type F or P
"""

class Cell:
    def __init__(self, position, state = 0):
        self.position = position                                    #(x, y)
        self.state = state
        self.weight = np.random.rand()                              # All cells must have a weight, so that the Agent will have a weight-based memory of the environment 

    def __str__(self):
            return 'Cell at {self.position}'.format(self=self)

class Consumable(Cell):
    def __init__(self, position, type, quantity):
        super().__init__(position)
        self.state = 2                  # Consumable state 
        self.type = type                # 'F' or 'P' 
        self.quantity = quantity        # Total quantity that could be consumed
        self.weight = np.random.rand()  # Weight on the cell matrix map
    
    def __str__(self):
        if self.type == 'P':
            return 'Poison at {self.position}'.format(self=self)
        elif self.type == 'F':
            return 'Food at {self.position}'.format(self=self)
    
    def consumed(self, q):
        self.quantity -= q
            
    def is_empty(self):
        if self.quantity <= 1:
            self.quantity = 0
            return True
        else:
            return False 

class Agent(Cell):
    def __init__(self, position, hp = 25):                     
        super().__init__(position)                                                               # Position is of the form (x, y)
        self.state = 1
        self.hp = hp
        self.sensors = [0.5, 0.5]                                                                # initialize sensors randomly as a probability summing up to 1. 
        self.factors = list(np.random.dirichlet(np.ones(3),size=1)[0])  
        self.sensors_weights = list(np.random.dirichlet(np.ones(2),size=1)[0])
        self.ext_factors = list(np.random.dirichlet(np.ones(3),size=1)[0])
        self.food_consumed = [0.001]
        self.poison_consumed = [0.001]
        self.empty_steps = 1

    def __str__(self):
            return 'Agent at {self.position}'.format(self=self)

    def get_neighbourhood(self, cells):  # cells should be passed as env.cells (the cells matrix, or the 2d space)
        nbh = []
        x, y = self.position[0], self.position[1]
        nbh.append(cells[x][y+1])   # North
        nbh.append(cells[x+1][y])   #East
        nbh.append(cells[x-1][y])   # West
        nbh.append(cells[x][y-1])   # South
        nbh.append(cells[x+1][y-1]) # SE
        nbh.append(cells[x-1][y-1]) # SW
        nbh.append(cells[x+1][y+1]) # NE
        nbh.append(cells[x-1][y+1]) # NW
        nbh = [n for n in nbh if not isinstance(n, int)]
        return nbh
    
    def get_extended_neighbourhood(self, cells):  # cells should be passed as env.cells (the cells matrix, or the 2d space)
        nbh = []
        x, y = self.position[0], self.position[1]
        nbh.append(cells[x][y+2])   # North
        nbh.append(cells[x+2][y])   #East
        nbh.append(cells[x-2][y])   # West
        nbh.append(cells[x][y-2])   # South
        nbh.append(cells[x+2][y-2]) # SE
        nbh.append(cells[x-2][y-2]) # SW
        nbh.append(cells[x+2][y+2]) # NE
        nbh.append(cells[x-2][y+2]) # NW
        nbh.append(cells[x-1][y+2]) 
        nbh.append(cells[x+1][y+2]) 
        nbh.append(cells[x+2][y+1])
        nbh.append(cells[x+2][y-1])
        nbh.append(cells[x-2][y+1])
        nbh.append(cells[x-2][y-1]) 
        nbh.append(cells[x-1][y-2])
        nbh.append(cells[x+1][y-2])     
        nbh = [n for n in nbh if not isinstance(n, int)]
        return nbh
    
    def inspect(self, ext_nbh):
        for cell in ext_nbh:
            if cell.state == 2:
                return True
            else:
                return False


    """
    decide_step function determines what the agent does next. 
    The agent takes into consideration its Moore neighbourhood and an additional extended neighbourhood (a Moore neighbourhood for the inward Moore neighbourhood). 
    The extended neighbourhood is checked if inspect returns True, which checks for the presence of Consumables. 
    Then, the weights map (each cell on the matrix environment has an associated weight) further inform the agent's decision.
    Finally, the choice is a function of weights and meta-weights associated with each option identified. 
    """
    def decide_step(self, cells, nbh, ext_nbh, inspect):
        options = [[], []]
        ext_options = [[], []]

        for i in range(len(nbh)):
            for j in range(len(cells[0])):
                for k in range(len(cells[1])):
                    if not isinstance(cells[j][k], int):
                        if cells[j][k].position == nbh[i].position and cells[j][k].state != 1:
                            options[0].append(cells[j][k].weight)
                            options[1].append(cells[j][k])

        for i in range(len(ext_nbh)):
            for j in range(len(cells[0])):
                for k in range(len(cells[1])):
                    if not isinstance(cells[j][k], int):
                        if cells[j][k].position == ext_nbh[i].position and cells[j][k].state != 1 and not isinstance(cells[j][k], int):
                                ext_options[0].append(cells[j][k].weight)
                                ext_options[1].append(cells[j][k])
        for i in range(len(options[1])):
            if options[1][i].state == 2:
                if options[1][i].type == 'F':
                    options[0][i] *= self.factors[0] 
                elif options[1][i].type == 'P':
                    options[0][i] *= self.factors[1]
            elif options[1][i].state == 0:
                options[0][i] *= self.factors[2]

        if inspect:
            for i in range(len(ext_options)):
                for j in range(len(options)):
                    if (ext_options[1][i].position[0] - 1 == options[1][j].position[0]) or (ext_options[1][i].position[0]+1 == options[1][j].position[0]) or (ext_options[1][i].position[1]+1 == options[1][j].position[1]) or (ext_options[1][i].position[1] == options[1][j].position[1] - 1)or (ext_options[1][i].position[0]+1 == options[1][j].position[1]):
                        if ext_options[1][i].state == 2:
                            if ext_options[1][i].type == 'F':
                                options[0][j] *= self.ext_factors[0] * self.factors[0]
                            if ext_options[1][i].type == 'P':
                                options[0][j] *= self.ext_factors[1] * self.factors[1]
                        elif ext_options[1][i].state == 0:
                                options[0][j] *= self.ext_factors[2] * self.factors[2]

        maxim = max(options[0])
        choice = options[1][options[0].index(maxim)] 

        # If the choice is not in options, raise an Error.  
        if choice not in nbh:
            raise ValueError(f"Expected Cell within neighbourhood, received {choice}.")
        
        return choice, options, ext_options
       
    def update_sensors_weights(self, act):
        self.sensors_weights = [act[0], act[1]]
        
    def update_sensors(self, act):
        self.update_sensors_weights(act)
        self.factors = [act[2], act[3], act[4]]
        self.ext_factors = [act[5], act[6], act[7]]  
    
    def consume_cell(self, cell):
        if cell.state == 0:
            pass
        elif cell.state == 2:
            if cell.type == 'F':
                self.hp += cell.quantity*self.sensors[0]*self.sensors_weights[0]
                ate = cell.quantity*self.sensors[0]*self.sensors_weights[0]
                self.food_consumed.append(ate)
            elif cell.type == 'P':
                self.hp -= cell.quantity*self.sensors[1]*self.sensors_weights[1]
                ate = cell.quantity**self.sensors[1]*self.sensors_weights[1]
                self.poison_consumed.append(abs(ate))
        if self.hp > 100:
            self.hp = 100
        return ate

