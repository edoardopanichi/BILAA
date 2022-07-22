import chess
import numpy as np
from stockfish import Stockfish
from monte_carlo_search_tree import MCTS
from IPython.display import clear_output

class chessboard:
    def __init__(self):
        self.board = chess.Board()
        
    def random_legal_move(self):
        if self.board.is_game_over() == False:
            legal_moves = list(self.board.legal_moves)
            
            rand_move = np.random.randint(len(legal_moves))
            self.board.push(legal_moves[rand_move])
            game_finished = False
                
        else: 
            print(self.board.outcome())
            game_finished = True
            
        return game_finished
    
class stockfish_eng:
    def __init__(self):
        # Ubuntu: /home/edoardo/Desktop/BILAA/Stockfish-master/src/stockfish
        # Mac: /Users/edoardo/Desktop/BILAA/Stockfish-master/src/stockfish
        self.engine = Stockfish(path="../Stockfish-master/src/stockfish")
        
    def play_best_move(self, board):
        self.engine.set_fen_position(board.fen())
        move = self.engine.get_best_move()
        
        return move
    
    def ELO_and_skill_value(self, elo, skill):
        # Allowing to reduce Stockfish level (apparently the minimum ELO that stockfish can play is 1350 - 
        # Quite high level!)
        limit_strength = {'UCI_LimitStrength': 'true'}
        self.engine._parameters.update(limit_strength)
        self.engine.set_elo_rating(elo)
        self.engine.set_skill_level(skill)
    
    def stockfish_vs_EA(self, agent, starting_elo, starting_skill, stockfish_is_white = True, mcst_epochs = 5, mcst_depth = 5):
        mcts = MCTS()
        model = agent.neural_network
                    
        def evaluation(input):
            pred = model(input.reshape(1, 8, 8, 12))
            return pred
        
        self.ELO_and_skill_value(starting_elo, starting_skill)
        print("questo si.")
        print("parameters: ", self.engine.get_parameters())
        
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
                starting_elo += 150
                starting_skill += 1
                self.ELO_and_skill_value(starting_elo, starting_skill)
                
            match += 1
            
        return starting_elo, starting_skill, board
        
        
        
        

    