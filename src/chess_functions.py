import chess
import numpy as np
from stockfish import Stockfish
from monte_carlo_search_tree import MCTS
from IPython.display import clear_output

    
def random_legal_move(board):
    if board.is_game_over() == False:
        legal_moves = list(board.legal_moves)
        
        rand_move = np.random.randint(len(legal_moves))
        board.push(legal_moves[rand_move])
        game_finished = False
            
    else: 
        print(board.outcome())
        game_finished = True
        
    return game_finished

def random_board_setup(board, moves_played = 25):
    position_ok = False
    while (position_ok == False): 
        for i in range(moves_played):
            if board.is_game_over() == False:
                legal_moves = list(board.legal_moves)
                
                rand_move = np.random.randint(len(legal_moves))
                board.push(legal_moves[rand_move])
            
            # If the previous move ended the game, we withdraw the move.       
            else: 
                board.pop()
                
        # If the position obtained allows further moves, then we stop the function. Otherwise we reset 
        # the board and we try again a new random position.  
        if (board.outcome() == None):  
            position_ok = True
        else:
            board = chess.Board()
    
class stockfish_eng:
    def __init__(self):
        # More depth makes stockfish better
        self.engine = Stockfish(path="../Stockfish-master/src/stockfish", depth=2)
        # engine2 = chess.engine.SimpleEngine.popen_uci(r"../Stockfish-master/src/stockfish")
        
    def play_best_move(self, board):
        self.engine.set_fen_position(board.fen())
        move = self.engine.get_best_move()
        
        return move
    
    def skill_value(self, skill):
        # Allowing to reduce Stockfish level (apparently the minimum Skill=0 that stockfish can play is 
        # quite high level!)
        # limit_strength = {'UCI_LimitStrength': 'true'}
        # self.engine._parameters.update(limit_strength)
        # self.engine.set_elo_rating(elo)
        self.engine.set_skill_level(skill)
    
    def stockfish_vs_EA(self, agent, starting_skill, stockfish_is_white = True, mcst_epochs = 5, mcst_depth = 5):
        mcts = MCTS()
        model = agent.neural_network
                    
        def evaluation(input):
            pred = model(input.reshape(1, 8, 8, 12))
            return pred
        
        self.skill_value(starting_skill)
        
        moves_list = [] # to store how many moves each match lasted
        white = 1
        moves = 0
        EA_lost = False
        match = 1
        
        while (EA_lost==False):
            clear_output()
            board = chess.Board()
            print("started match", str(match), "!")
            
            while((not board.is_game_over())):       
                
                # According to the turns, either stockfish or the EA play a move
                if stockfish_is_white:
                    if white:
                        result = self.play_best_move(board)
                        print("stockfish (white) plays: ", result)
                        board.push_san(result)
                    else:
                        result, _ = mcts.simple_mcst(board, evaluation, epochs = mcst_epochs, depth = mcst_depth)
                        print("EA (black) plays: ", result)   
                        board.push(result) 
                else:
                    if white:
                        result, _ = mcts.simple_mcst(board, evaluation, epochs = mcst_epochs, depth = mcst_depth)
                        print("EA (white) plays: ", result)   
                        board.push(result) 
                    else:
                        result = self.play_best_move(board)
                        print("stockfish (black) plays: ", result)
                        board.push_san(result) 
                
                # the operator ^= Performs Bitwise OR on operands and assign value to left operand. This means that the 
                # value of white is flipped between 0 and 1 after each move.
                white ^= 1
                moves += 1
                
            # 'board.outcome().winner' is true if the game has been won by white
            if board.outcome().winner and stockfish_is_white: # i.e. Stockfish won
                    print("\n STOCKFISH WINS WITH WHITE")
                    EA_lost = True
                    
            elif (board.outcome().winner == False) and (not stockfish_is_white): # i.e. Stockfish won
                print("\n STOCKFISH WINS WITH BLACK")
                EA_lost = True    
            
            else: # i.e EA won or draw
                # The maximum skill level for Stockfish is 21
                if starting_skill < 20:
                    starting_skill += 1
                    self.skill_value(starting_skill)
                else:
                    return starting_skill, board
                
            match += 1
            moves_list.append(moves)
            
        return starting_skill, board, moves_list
    
    # It generates a list of the top n moves and a list of the evaluation of the final position for each 
    # of the moves.
    def top_moves(self, board, num_of_moves = 3):
        
        self.engine.set_fen_position(board.fen())
        
        # best_moves is a list of dictionaries
        best_moves = self.engine.get_top_moves(num_of_moves)
        
        top_moves = []
        centipawn = [] # The centipawn is the unit of measure used in chess as measure of the advantage.
        # A centipawn is equal to 1/100 of a pawn. Therefore, 100 centipawns = 1 pawn. 
        for move in best_moves:
            top_moves.append(move["Move"])
            centipawn.append(move["Centipawn"])
            
        return top_moves, centipawn
        
        
        
        

    