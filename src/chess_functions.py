import chess
import numpy as np

class chessboard:
    def __init__(self):
        self.board = chess.Board()
        
    def random_legal_move(self):
        legal_moves = list(self.board.legal_moves)
        rand_move = np.random.randint(len(legal_moves))
        self.board.push(legal_moves[rand_move])
        

    