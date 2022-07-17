import chess
from evaluation_class import evaluator
from monte_carlo_search_tree import MCTS
import random

# Using the fitness function we can determine which agent should ensure better performances in terms of 
# chances of winning a match.
def fitness(agents):
    
    # Initialization of the class that contains the monte-carlo search tree
    mcts = MCTS()
    
    # Each agent will play a match with another random agent. So averagely, each agent will play two matches.
    for agent in range(len(agents) - 1):
        game = []
        board = chess.Board()
        
        # Selecting two agents to play a match and update their fitness score
        player_1 = agents[agent]
        player_2 = random.choice(agents)
        
        player_1_idx = agent
        player_2_idx = agents.index(player_2)

        print('Game Started between Agent', player_1_idx, 'and Agent', player_2_idx)
        counter = 0
        
        while counter < 100 and board.is_game_over() == False:
            model = player_1.neural_network
            
            def evaluation(input):
                pred = model(input.reshape(1, 8, 8, 12))
                return pred
            
            # Move for player 1
            move =  mcts.simple_mcst(board, evaluation, epochs = 5, depth = 5)
            game.append(move)
            
            # Move for player 2
            model = player_2.neural_network
            move =  mcts.simple_mcst(board, evaluation, epochs = 5, depth = 5)
            game.append(move)
            
            counter += 1

        # For each agent we save the list of moves played in the game
        agents[player_1_idx].game = game
        agents[player_2_idx].game = game

        if board.is_checkmate:
            # The counter helps us understand if the winner was player 1 or player 2.
            if counter % 2 == 0:
                agents[player_1_idx].fitness *= 1.5
                agents[player_2_idx].fitness *= 0.8
                
            else:
                agents[player_2_idx].fitness *= 1.5
                agents[player_1_idx].fitness *= 0.8
                
    return agents

