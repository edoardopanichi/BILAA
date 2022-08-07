import chess
from evaluation_class import evaluator
from fitness_function import *
from evolutionary_algorithm import genetic_algorithm, Agent_copy
import time
from store_load_models import load_store_models



# with this parameters crashes at gen 6 due to lack of memory
pop_size = 18
generations = 6
mcst_epochs = 8
mcst_depth = 3

# with this parameters crashes at gen 14 due to lack of memory
pop_size = 20
generations = 5
mcst_epochs = 8
mcst_depth = 2


# We define an object of the class evaluator to generate the model of the NN that evaluates the board.
eval_model = evaluator()
# Instance of the fitness function, used to define the fitness of the agents.
fitness_func = fitness
# Core of the code: the evolutionary algorithm that defines the evolution generation after generation.
ga = genetic_algorithm()


start_time = time.time() # To measure the time needed to train the models.
agent, loss, gen_wins, gen_moves = ga.execute(fitness_func, eval_model.simple_eval_model(), prev_agent=None, pop_size = pop_size, generations = generations, mcst_epochs = mcst_epochs, mcst_depth = mcst_depth)

# Updating some information about the set parameters to obtain "agent". These are useful for some data analysis and 
# parameters adjustment later on.
agent.training_time = (time.time() - start_time)/3600 # (converting it in hours)
agent.pop_size = pop_size
agent.generations = generations
agent.mcst_epochs = mcst_epochs
agent.mcst_depth = mcst_depth
agent.description = str("This model does not shown any particular skill")
agent.loss_progression = loss
agent.gen_wins = gen_wins
agent.gen_moves = gen_moves

# Copying the agent in a format that can be saved by the pickle module in one of the next cells
agent_copy = Agent_copy()
agent_copy.copy_agent(agent)

trained_models = load_store_models(file_empty=False, agent_copy=agent_copy, store=str("yes"))