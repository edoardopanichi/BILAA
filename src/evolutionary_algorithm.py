import random
import numpy as np
from IPython.display import clear_output
from keras.models import clone_model
import tensorflow as tf

# The following is a command to suppress some output when using tensorflow library
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def sigmoid(x):
    return 1/(1+np.exp(-x))

class genetic_algorithm:
        
    def execute(self, fitness, model, pop_size = 10, generations = 100, threshold = 1000):
        
        # The class agent allows us to define a player with its own model of the NN for the evaluation 
        # that it is used in the Monte Carlo search tree to evaluate a given position of the board.
        class Agent:
            def __init__(self, model):
                # clone_model() by keras: Clone a Functional or Sequential Model instance
                self.neural_network = clone_model(model)
                self.fitness = 100
                self.game = None
                
            # The __str__ method is called when the following functions are invoked on the object and 
            # return a string: print(), str()    
            def __str__(self):
                    return 'Loss: ' + str(self.fitness)
      
            def apply_weights(self, weights):
                self.neural_network.set_weights(weights)
            
        # Generation of the desired amount of agents with the same model of NN.   
        def generate_agents(population, network):
            return [Agent(model) for _ in range(population)]
        
        def selection(agents):
            agents = sorted(agents, key=lambda agent: agent.fitness, reverse=True)
            print('\n'.join(map(str, agents)))
            agents = agents[:int(0.2 * len(agents))]
            return agents
        
        def unflatten(flattened, shapes):
            new_array = []
            index = 0
            for shape in shapes:
                size = np.product(shape)
                new_array.append(flattened[index : index + size].reshape(shape))
                index += size
            return np.array(new_array)
        
        def crossover(agents,network,pop_size):
            offspring = []
            for _ in range((pop_size - len(agents)) // 2):
                parent1 = random.choice(agents)
                parent2 = random.choice(agents)
                child1 = Agent(network)
                child2 = Agent(network)
                
                
                shapes = [a.shape for a in parent1.neural_network.get_weights()]
                genes1 = np.concatenate([a.flatten() for a in parent1.neural_network.get_weights()])
                genes2 = np.concatenate([a.flatten() for a in parent2.neural_network.get_weights()])
                split = random.randint(0,len(genes1)-1)

                child1_genes = np.array(genes1[0:split].tolist() + genes2[split:].tolist())
                child2_genes = np.array(genes1[0:split].tolist() + genes2[split:].tolist())
                child1_genes = unflatten(child1_genes,shapes)
                child2_genes = unflatten(child2_genes,shapes)
            
                
                child1.apply_weights(list(child1_genes))
                child2.apply_weights(list(child2_genes))
                
                
                offspring.append(child1)
                offspring.append(child2)
            agents.extend(offspring)
            return agents
        
        def mutation(agents):
            for agent in agents:
                if random.uniform(0.0, 1.0) <= 0.1:
                    weights = agent.neural_network.get_weights()
                    shapes = [a.shape for a in weights]

                    flattened = np.concatenate([a.flatten() for a in weights])
                    randint = random.randint(0,len(flattened)-1)
                    flattened[randint] = np.random.randn()

                    newarray = unflatten(flattened,shapes)
                    agent.apply_weights(newarray)
            return agents
        
        loss = []
        for i in range(generations):
            print('Generation',str(i),':')
            agents = generate_agents(pop_size,model)
            agents = fitness(agents)
            agents = selection(agents)
            agents = crossover(agents,model,pop_size)
            agents = mutation(agents)
            agents = fitness(agents)
            loss.append(agents[0].fitness)
            if any(agent.fitness > threshold for agent in agents):
                print('Threshold met at generation '+str(i)+' !')
                
            if i % 100:
                clear_output()
                
        return agents[0],loss