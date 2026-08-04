import chess
import numpy as np
from evaluation_class import evaluator, prediction_to_values
from monte_carlo_search_tree import MCTS
import random
import time


DEFAULT_OPENING_SEQUENCES = (
    (),
    ("e2e4", "e7e5", "g1f3", "b8c6"),
    ("d2d4", "d7d5", "c2c4", "e7e6"),
    ("c2c4", "e7e5", "g1f3", "b8c6"),
    ("g1f3", "d7d5", "e2e4", "e7e6"),
    ("b1c3", "g8f6", "e2e4", "d7d5"),
    ("e2e4", "c7c5", "g1f3", "d7d6"),
    ("d2d4", "g8f6", "c2c4", "g7g6"),
)


def _opening_board(sequence):
    board = chess.Board()
    for move in sequence:
        board.push_uci(move)
    return board


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
    opening_sequences=None,
    evaluation_seed=None,
):
    if len(agents) < 2 or len(agents) % 2 != 0:
        raise ValueError("Balanced fitness requires an even population of at least two agents.")
    if games_per_agent < 2 or games_per_agent % 2 != 0:
        raise ValueError("games_per_agent must be an even number of at least two.")

    mcts = MCTS(state_features=state_features)
    if opening_sequences is None:
        opening_sequences = ((),)
    if not opening_sequences:
        raise ValueError("opening_sequences must contain at least one sequence.")

    stats = {
        "games": 0,
        "decisive_games": 0,
        "white_wins": 0,
        "black_wins": 0,
        "draws": 0,
        "total_plies": 0,
    }
    score_totals = {id(agent): 0.0 for agent in agents}
    game_counts = {id(agent): 0 for agent in agents}
    schedule_rng = random.Random(evaluation_seed) if evaluation_seed is not None else None

    def add_score(agent, value):
        if scoring == "robust":
            score_totals[id(agent)] += value
            game_counts[id(agent)] += 1
        else:
            agent.fitness += value

    def play_match(white_agent, black_agent, opening_sequence, match_seed=None):
        board = _opening_board(opening_sequence)
        rng = random.Random(match_seed) if match_seed is not None else None
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
                rng=rng,
            )
            board.push(move)
            counter += 1

        stats["games"] += 1
        stats["total_plies"] += counter
        outcome = board.outcome()
        if scoring == "legacy":
            if outcome is not None and outcome.winner is not None:
                if outcome.winner == chess.WHITE:
                    white_agent.fitness *= 1.5
                    black_agent.fitness *= 0.8
                else:
                    black_agent.fitness *= 1.5
                    white_agent.fitness *= 0.8
                stats["decisive_games"] += 1
                if outcome.winner == chess.WHITE:
                    stats["white_wins"] += 1
                else:
                    stats["black_wins"] += 1
                return 1
            stats["draws"] += 1
            return 0

        if outcome is not None and outcome.winner == chess.WHITE:
            add_score(white_agent, 1.0)
            add_score(black_agent, -1.0)
            stats["decisive_games"] += 1
            stats["white_wins"] += 1
            return 1
        if outcome is not None and outcome.winner == chess.BLACK:
            add_score(black_agent, 1.0)
            add_score(white_agent, -1.0)
            stats["decisive_games"] += 1
            stats["black_wins"] += 1
            return 1

        stats["draws"] += 1
        margin = draw_material_weight * np.tanh(_material_balance(board) / 9.0)
        add_score(white_agent, float(margin))
        add_score(black_agent, -float(margin))
        return 0

    decisive_games = 0
    for round_index in range(games_per_agent // 2):
        order = list(range(len(agents)))
        if schedule_rng is None:
            random.shuffle(order)
        else:
            schedule_rng.shuffle(order)
        if verbose:
            print("\nBALANCED ROUND", round_index)

        for pair_index in range(0, len(order), 2):
            first = agents[order[pair_index]]
            second = agents[order[pair_index + 1]]
            opening_index = round_index * (len(order) // 2) + pair_index // 2
            opening_sequence = opening_sequences[opening_index % len(opening_sequences)]
            match_seed = None
            if evaluation_seed is not None:
                match_seed = evaluation_seed + opening_index * 2
            decisive_games += play_match(first, second, opening_sequence, match_seed)
            decisive_games += play_match(second, first, opening_sequence, match_seed)

    if scoring == "robust":
        for agent in agents:
            average_score = score_totals[id(agent)] / game_counts[id(agent)]
            agent.fitness = 100.0 + 10.0 * average_score

    stats["average_plies"] = (
        stats["total_plies"] / stats["games"] if stats["games"] else 0.0
    )
    _balanced_fitness.last_metrics = stats

    return agents, decisive_games


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
    opening_sequences=None,
    evaluation_seed=None,
):
    if schedule == "balanced" or scoring != "legacy" or state_features or root_perspective:
        result = _balanced_fitness(
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
            opening_sequences,
            evaluation_seed,
        )
        fitness.last_metrics = dict(_balanced_fitness.last_metrics)
        return result
    
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
                
    fitness.last_metrics = {}
    return agents, wins

