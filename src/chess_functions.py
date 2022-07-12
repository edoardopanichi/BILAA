import chess
import numpy as np
from stockfish import Stockfish

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
        

    