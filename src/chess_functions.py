import chess
import numpy as np

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
        

    