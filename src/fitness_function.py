import chess
import numpy as np
from evaluation_class import evaluator, prediction_to_values
from monte_carlo_search_tree import MCTS
import random
import time


def _material_balance(board):
    values = {
        chess.PAWN: 1.0,
        chess.KNIGHT: 3.0,
        chess.BISHOP: 3.0,
        chess.ROOK: 5.0,
        chess.QUEEN: 9.0,
        chess.KING: 0.0,
    }
    balance = 0.0
    for piece_type, value in values.items():
        balance += value * len(board.pieces(piece_type, chess.WHITE))
        balance -= value * len(board.pieces(piece_type, chess.BLACK))
    return balance


def _balanced_fitness(
    agents,
    mcst_epochs,
    mcst_depth,
    max_plies,
    games_per_agent,
    state_features,
    root_perspective,
    terminal_reward,
    draw_material_weight,
    random_ties,
    scoring,
    verbose,
):
    if len(agents) < 2 or len(agents) % 2 != 0:
        raise ValueError("Balanced fitness requires an even population of at least two agents.")
    if games_per_agent < 2 or games_per_agent % 2 != 0:
        raise ValueError("games_per_agent must be an even number of at least two.")

    mcts = MCTS(state_features=state_features)
    wins = 0

    def play_match(white_agent, black_agent):
        board = chess.Board()
        counter = 0
        while counter < max_plies and not board.is_game_over():
            current_agent = white_agent if board.turn == chess.WHITE else black_agent
            model = current_agent.neural_network

            def evaluation(input_batch):
                batch = np.asarray(input_batch)
                if batch.ndim == 3:
                    batch = batch[None, ...]
                return prediction_to_values(model(batch))

            move, _ = mcts.simple_mcst(
                board,
                evaluation,
                epochs=mcst_epochs,
                depth=mcst_depth,
                root_perspective=root_perspective,
                terminal_reward=terminal_reward,
                random_ties=random_ties,
            )
            board.push(move)
            counter += 1

        outcome = board.outcome()
        if scoring == "legacy":
            if outcome is not None and outcome.winner is not None:
                if outcome.winner == chess.WHITE:
                    white_agent.fitness *= 1.5
                    black_agent.fitness *= 0.8
                else:
                    black_agent.fitness *= 1.5
                    white_agent.fitness *= 0.8
                return 1
            return 0

        if outcome is not None and outcome.winner == chess.WHITE:
            white_agent.fitness += 1.0
            black_agent.fitness -= 1.0
            return 1
        if outcome is not None and outcome.winner == chess.BLACK:
            black_agent.fitness += 1.0
            white_agent.fitness -= 1.0
            return 1

        margin = draw_material_weight * np.tanh(_material_balance(board) / 9.0)
        white_agent.fitness += float(margin)
        black_agent.fitness -= float(margin)
        return 0

    for round_index in range(games_per_agent // 2):
        order = list(range(len(agents)))
        random.shuffle(order)
        if verbose:
            print("\nBALANCED ROUND", round_index)

        for pair_index in range(0, len(order), 2):
            first = agents[order[pair_index]]
            second = agents[order[pair_index + 1]]
            wins += play_match(first, second)
            wins += play_match(second, first)

    return agents, wins


# Using the fitness function we can determine which agent should ensure better performances in terms of 
# chances of winning a match.
def fitness(
    agents,
    mcst_epochs,
    mcst_depth,
    max_plies=300,
    verbose=True,
    schedule="legacy",
    scoring="legacy",
    games_per_agent=2,
    state_features=False,
    root_perspective=False,
    terminal_reward=None,
    draw_material_weight=0.25,
    random_ties=False,
):
    if schedule == "balanced" or scoring != "legacy" or state_features or root_perspective:
        return _balanced_fitness(
            agents,
            mcst_epochs,
            mcst_depth,
            max_plies,
            games_per_agent,
            state_features,
            root_perspective,
            terminal_reward,
            draw_material_weight,
            random_ties,
            scoring,
            verbose,
        )
    
    # Initialization of the class that contains the monte-carlo search tree
    mcts = MCTS()
    # Variable to account for the number of wins among the agents
    wins = 0
    
    # Each agent faces 1/3 of the population to determine which agent is the fittest
    for j in range(round(len(agents)/3)):
        if verbose:
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

            if verbose:
                print('Game Started between Agent', player_1_idx, 'and Agent', player_2_idx)
            counter = 0
            
            while counter < max_plies and board.is_game_over() == False:
                model = player_1.neural_network
                
                def evaluation(input):
                    batch = np.asarray(input)
                    if batch.ndim == 3:
                        batch = batch[None, ...]
                    return prediction_to_values(model(batch))
                
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
                if verbose:
                    print("outcome of the match: ", board.outcome())
                
                if(board.outcome().winner or (board.outcome().winner == False)):
                # The counter helps us understand if the winner was player 1 or player 2.
                    if verbose:
                        print("Updating fitness...")
                    if (counter % 2) != 0:
                        agents[player_1_idx].fitness *= 1.5
                        agents[player_2_idx].fitness *= 0.8
                        
                    else:
                        agents[player_2_idx].fitness *= 1.5
                        agents[player_1_idx].fitness *= 0.8
                    
                    wins += 1
                    
            if verbose:
                print("This game took (in sec):", time.time()-start_time, ", and counter is:", str(counter), "\n")
                
    return agents, wins

