from audioop import avg
import random
import numpy as np
from IPython.display import clear_output
from keras.models import clone_model
import tensorflow as tf
from chess_functions import stockfish_eng

# The following is a command to suppress some output when using tensorflow library
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class genetic_algorithm:
        
    def execute(self, fitness, model, prev_agents = None, pop_size = 10, generations = 100, mcst_epochs = 5, mcst_depth = 5):
        
        # The class agent allows us to define a player with its own model of the NN for the evaluation 
        # that it is used in the Monte Carlo search tree to evaluate a given position of the board.
        class Agent:
            def __init__(self, model):
                # clone_model() by keras: Clone a Functional or Sequential Model instance generating new 
                # random weights.
                self.neural_network = clone_model(model)
                self.fitness = 100
                self.game = None
                
                # The following variables are filled only for the final agent obtained after the training. In 
                # this way the instance Agent contains all the information to reproduce similar results.
                self.training_time = None
                self.pop_size = None
                self.generations = None
                self.mcst_epochs = None
                self.mcst_depth = None
                self.description = None
                self.loss_progression = None
                self.gen_wins = None # with this list we keep track of how many games are won in each 
                # generation
                self.gen_moves = None
                
            # The __str__ method is called when the following functions are invoked on the object and 
            # return a string: print(), str()    
            def __str__(self):
                    return 'Loss: ' + str(self.fitness)
      
            def apply_weights(self, weights):
                self.neural_network.set_weights(weights)
            
        # Generation of the desired amount of agents with the same model of NN.   
        def generate_agents(population):
            if population == 0:
                return []
            
            return [Agent(model) for _ in range(population)]
        
        # First way to develop the EA is through selection: given a certain population, only the fittest
        # are preserved.
        def selection(agents):
            # sorting according to the fitness value of the agents, starting from the greater values
            agents = sorted(agents, key=lambda agent: agent.fitness, reverse=True)
            # printing the fitness of each agent
            print("The loss of the agents of the previous generation was: \n")
            print('\n'.join(map(str, agents)))
            # Out of the n agents we keep only 20%, in particular the first 20% of the list where the 
            # fittest are kept.
            agents = agents[:int(0.2 * len(agents))]
            
            return agents
        
        def unflatten(flattened, shapes):
            # "shapes" is a list where each element is the shape of a layer of the NN.
            # "flattened" is a gene sequence of a new offspring. It has to be reordered as list where each 
            # element is contains the weights of a layer of the model.
            new_array = []
            index = 0
            for shape in shapes:
                # "size" indicates how many element of "flattened" forms a layer of weights for the NN.
                size = np.product(shape)
                new_array.append(flattened[index : index + size].reshape(shape))
                # "index" has to be update to select the elements of the next layer
                index += size
            return np.array(new_array, dtype=object)
        
        # The second way to develop the EA is through the crossover step: the genes of fittest samples of
        # the population are mixed to obtain new agents with characteristic coming from 2 previous fit 
        # agents.
        # Here is a step-by-step explanation:
            # 1. Their weights are flattened so that values can be changed.
            # 2. A random intersection point is found. This point is where the genetic information of one
            # parent ends, and where the genetic information of one parent begins.
            # 3. The genes of the parents are joined and a new child agent then holds the weight created 
            # by this operation.
        def crossover(agents, network, pop_size):
            # The agents entering in this function have already been selected, i.e. they are the fittest 20%
            # of the previous generation.
            offspring = []
            
            # Given a the population size (pop_size), 80% of the new generation is obtained with crossover. 
            # Each crossover generates two new agents.
            for _ in range((pop_size - len(agents)) // 2):
                parent1 = random.choice(agents)
                parent2 = random.choice(agents)
                # Generation of two new agents, giving as input a blank NN model
                child1 = Agent(network)
                child2 = Agent(network)
                
                # get_weights(): Returns the current weights of the layer, as NumPy arrays. "shapes" is a list
                # where each element is the shape of a layer of the NN.
                shapes = [a.shape for a in parent1.neural_network.get_weights()]
                # genes1 and genes2 are a long sequence containing the weights of the NN for the parent1 and
                # parent2.
                genes1 = np.concatenate([a.flatten() for a in parent1.neural_network.get_weights()])
                genes2 = np.concatenate([a.flatten() for a in parent2.neural_network.get_weights()])
                # Picking a random point to divide the genes between parent1 and parent2
                split = random.randint(0, len(genes1)-1)

                child1_genes = np.array(genes1[0:split].tolist() + genes2[split:].tolist())
                child2_genes = np.array(genes2[0:split].tolist() + genes1[split:].tolist())
                # To use the child_genes as weights of the NN we need to structure them as list where each 
                # element is contains the weights of a layer of the model. To do so we can exploit the 
                # function unflatten.
                child1_genes = unflatten(child1_genes, shapes)
                child2_genes = unflatten(child2_genes, shapes)
                
                child1.apply_weights(list(child1_genes))
                child2.apply_weights(list(child2_genes))
                
                offspring.append(child1)
                offspring.append(child2)
            
            # The extend() method modifies the original list adding the new elements to the list.
            agents.extend(offspring)
            # agents now contains the 20% selected + the new crossover agents.
            return agents
        
        # Mutation receives agents that are already the new generation, i.e. agents after the function 
        # "selection" and "crossover".
        def mutation(agents):
            for agent in agents:
                # A mutation happens with a 10% probability
                if random.uniform(0.0, 1.0) <= 0.1:
                    weights = agent.neural_network.get_weights()
                    shapes = [a.shape for a in weights]

                    flattened = np.concatenate([a.flatten() for a in weights])
                    # selection of a random index for the weights. This index is used to pick the weight to 
                    # mutate.
                    randint = random.randint(0, len(flattened)-1)
                    flattened[randint] = np.random.randn()

                    new_array = unflatten(flattened, shapes)
                    agent.apply_weights(new_array)
            return agents
        
        # Every new generation, 20% of the agents are obtained from the previous generation. For these onces, 
        # we need to set the fitness back to zero to ensure a common starting point with the other agents.
        def reset_fitness(agents):
            for agent in agents:
                agent.fitness = 100
                
            return agents
        
        def moves_against_stockfish(agent, mcst_epochs, mcst_depth):
            engine = stockfish_eng()
            moves = []
            
            # We let the best agent of the generation play 10 games against stockfish to understand in how 
            # many moves the game ends
            for i in range(10):
                _, _, num_moves = engine.stockfish_vs_EA(agent, starting_skill=0, stockfish_is_white=True, mcst_epochs=mcst_epochs, mcst_depth=mcst_depth)
                
                # if the list num_moves is empty means that EA has won against stockfish
                if not num_moves:
                    # if EA wins we set num_moves to basically infinity. In this way reading the average
                    # number of moves in the matches will be clear that EA won at least once.
                    num_moves = [100000]
                    
                moves.extend(num_moves)
                
            avg_moves = sum(moves) / len(moves)
                
            return avg_moves
        
        
        
        loss = [] # list to track the improvements generation after generation.
        gen_wins = [] # list to track how many matches ends with a win in each generation
        gen_moves = []
        
        for i in range(generations):
            print('\nGeneration', str(i), ':')
            # In the first iteration we generate the starting agents and we evaluate their fitness score 
            if i == 0:
                if prev_agents == None:
                    agents = generate_agents(pop_size)  
                # If we start from a already trained agent we generate one less agent
                else:
                    num_prev_agents = len(prev_agents)
                    agents = generate_agents(pop_size - num_prev_agents)
                    
                    for prev_agent in prev_agents:
                        agents.append(prev_agent)  
                
                for agent in agents:
                    print(agent.fitness) 
                
                # Reset the fitness in case there are some pre-trained agents
                agents = reset_fitness(agents) 
                
                agents, wins = fitness(agents, mcst_epochs, mcst_depth)
                
            # For all the generation that are not the first, we start generating the new offspring, then we 
            # evaluate the fitness score.
            else:
                agents = selection(agents)
                agents = reset_fitness(agents)
                agents = crossover(agents, model, pop_size)
                agents = mutation(agents)
                agents, wins = fitness(agents, mcst_epochs, mcst_depth)
            
            # sorting according to the fitness value of the agents, starting from the greater values
            agents = sorted(agents, key=lambda agent: agent.fitness, reverse=True)  
            # "agents" are ordered from the fittest to the least fit. Hence "agent[0]" is the the best agent 
            # of the generation.   
            avg_moves = moves_against_stockfish(agents[0], mcst_epochs, mcst_depth)
            
            loss.append(agents[0].fitness)
            gen_wins.append(wins)
            gen_moves.append(avg_moves)
            
            if i % 100:
                clear_output()
                
        return agents[0], loss, gen_wins, gen_moves
    
# This class is used to save on external files the trained models   
class Agent_copy:
        def __init__(self):
            self.neural_network =  None
            self.fitness = None
            self.game = None
            # The following variables are filled only for the final agent obtained after the training. In 
            # this way the instance Agent contains all the information to reproduce similar results.
            self.training_time = None
            self.pop_size = None
            self.generations = None
            self.mcst_epochs = None
            self.mcst_depth = None
            self.description = None
            self.loss_progression = None
            self.gen_wins = None
            self.gen_moves = None
            
        def copy_agent(self, agent):
            self.neural_network = agent.neural_network
            self.fitness = agent.fitness
            self.game = agent.game
            self.training_time = agent.training_time
            self.pop_size = agent.pop_size
            self.generations = agent.generations
            self.mcst_epochs = agent.mcst_epochs
            self.mcst_depth = agent.mcst_depth
            self.description = agent.description
            self.loss_progression = agent.loss_progression
            self.gen_wins = agent.gen_wins
            self.gen_moves = agent.gen_moves
        
        # This function is needed if in the first generation a mutation happens to an agent developed in a 
        # previous run, then it is saved in form of agent copy.    
        def apply_weights(self, weights):
                self.neural_network.set_weights(weights)