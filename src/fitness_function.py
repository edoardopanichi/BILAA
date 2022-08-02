import chess
from evaluation_class import evaluator
from monte_carlo_search_tree import MCTS
import random
import time

# Using the fitness function we can determine which agent should ensure better performances in terms of 
# chances of winning a match.
def fitness(agents, mcst_epochs, mcst_depth):
    
    # Initialization of the class that contains the monte-carlo search tree
    mcts = MCTS()
    # Variable to account for the number of wins among the agents
    wins = 0
    
    # Each agent faces 1/3 of the population to determine which agent is the fittest
    for j in range(round(len(agents)/3)):
        print("\nROUND", j)
        
        # Each agent will play a match with another random agent. 
        for agent in range(len(agents)):
            start_time = time.time()
            game = []
            board = chess.Board()
            
            # Selecting two agents to play a match and update their fitness score
            player_1 = agents[agent]
            player_2 = random.choice(agents)
            
            # Player 2 has to be different from Player 1 
            while (player_1 == player_2):
                player_2 = random.choice(agents)
            
            player_1_idx = agent
            player_2_idx = agents.index(player_2)

            print('Game Started between Agent', player_1_idx, 'and Agent', player_2_idx)
            counter = 0
            
            while counter < 300 and board.is_game_over() == False:
                model = player_1.neural_network
                
                def evaluation(input):
                    pred = model(input.reshape(1, 8, 8, 12))
                    return pred
                
                # Move for player 1
                move, _ = mcts.simple_mcst(board, evaluation, epochs = mcst_epochs, depth = mcst_depth)
                board.push(move)
                game.append(move)
                counter += 1
                
                # Move for player 2
                # We need to check again if the match is over or not. It might be that the previous move ended the
                # game.
                if (board.is_game_over() == False):
                    model = player_2.neural_network
                    move, _ = mcts.simple_mcst(board, evaluation, epochs = mcst_epochs, depth = mcst_depth)
                    game.append(move)
                    board.push(move)
                    
                    counter += 1

            # For each agent we save the list of moves played in the game
            # agents[player_1_idx].game = game
            # agents[player_2_idx].game = game

            # If one of the two agent won, we update the fitness scores.
            if (board.outcome()):
                print("outcome of the match: ", board.outcome())
                
                if(board.outcome().winner or (board.outcome().winner == False)):
                # The counter helps us understand if the winner was player 1 or player 2.
                    print("Updating fitness...")
                    if (counter % 2) != 0:
                        agents[player_1_idx].fitness *= 1.5
                        agents[player_2_idx].fitness *= 0.8
                        
                    else:
                        agents[player_2_idx].fitness *= 1.5
                        agents[player_1_idx].fitness *= 0.8
                    
                    wins += 1
                    
            print("one game takes:", time.time()-start_time, ", and counter is:", str(counter), "\n")
                
    return agents, wins

