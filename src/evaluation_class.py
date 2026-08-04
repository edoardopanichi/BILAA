import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import chess
import numpy as np
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input


def prediction_to_values(prediction):
    """Convert a Keras backend prediction into a one-dimensional NumPy array."""
    if hasattr(prediction, "detach"):
        prediction = prediction.detach()
    if hasattr(prediction, "cpu"):
        prediction = prediction.cpu()
    if hasattr(prediction, "numpy"):
        prediction = prediction.numpy()
    return np.asarray(prediction).reshape(-1)


def prediction_to_float(prediction):
    """Convert a Keras backend scalar prediction into a Python float."""
    return float(prediction_to_values(prediction)[0])

# Simple NN model to evaluate a position on the board. The input shape is 8x8x12 because a chess game uses a 
# 8x8 board and there are 12 different possible pieces 8: 6 whites and 6 blacks (King, Queen, Rook, Bishop, 
# Knight, Pawn)
class evaluator:
    def __init__(self, include_state=False):
        self.include_state = include_state
        self.channels = 18 if include_state else 12
        
        self.chess_dict = {
                            'p' : [1,0,0,0,0,0,0,0,0,0,0,0],
                            'P' : [0,0,0,0,0,0,1,0,0,0,0,0],
                            'n' : [0,1,0,0,0,0,0,0,0,0,0,0],
                            'N' : [0,0,0,0,0,0,0,1,0,0,0,0],
                            'b' : [0,0,1,0,0,0,0,0,0,0,0,0],
                            'B' : [0,0,0,0,0,0,0,0,1,0,0,0],
                            'r' : [0,0,0,1,0,0,0,0,0,0,0,0],
                            'R' : [0,0,0,0,0,0,0,0,0,1,0,0],
                            'q' : [0,0,0,0,1,0,0,0,0,0,0,0],
                            'Q' : [0,0,0,0,0,0,0,0,0,0,1,0],
                            'k' : [0,0,0,0,0,1,0,0,0,0,0,0],
                            'K' : [0,0,0,0,0,0,0,0,0,0,0,1],
                            '.' : [0,0,0,0,0,0,0,0,0,0,0,0],
                            }           
    
    # Simple NN that takes has input information about the board set up, and produce an single output that 
    # evaluate numerically the position of the board.
    def simple_eval_model(self, image_shape=None, architecture="local"):
        if image_shape is None:
            image_shape = (8, 8, self.channels)

        if architecture == "board":
            model = Sequential([
                Input(shape=image_shape),
                Flatten(),
                Dense(32, activation="relu"),
                Dense(16, activation="relu"),
                Dense(1),
            ])
        else:
            model = Sequential([
                Input(shape=image_shape),
                Dense(10),
                Dense(10, activation='relu'),
                Flatten(),
                Dense(1),
            ])
        
        return model 
    
    # If we want to analyze a given board position we need to translate the board into data that can be easily
    # used by a NN. For this purpose we will use the chess_dict seen above that defined the notion for each 
    # piece.
    def translate(self, board): 
        # Extended Position Description (EPD) describes a chess position similar to the Forsyth-Edwards 
        # Notation (FEN). Unlike FEN, EPD is designed to be expandable by the addition of new operations. 
        pgn = board.epd()
        foo = []  
        # The split() method splits a string into a list. You can specify the separator, default separator 
        # is any whitespace.
        pieces = pgn.split(" ", 1)[0]
        rows = pieces.split("/")
        
        for row in rows:
            foo2 = []  
            for thing in row:
                if thing.isdigit():
                    for i in range(0, int(thing)):
                        foo2.append(self.chess_dict['.'])
                else:
                    foo2.append(self.chess_dict[thing])
            foo.append(foo2)
            
        # foo will be a list of 8x8 lists. Where each list is 12 digits long. Each list contains information 
        # about 1 square of the board.
        position = np.asarray(foo, dtype=np.float32)
        if not self.include_state:
            return position

        state = np.zeros((8, 8, 6), dtype=np.float32)
        state[:, :, 0] = 1.0 if board.turn == chess.WHITE else -1.0
        state[:, :, 1] = float(board.has_kingside_castling_rights(chess.WHITE))
        state[:, :, 2] = float(board.has_queenside_castling_rights(chess.WHITE))
        state[:, :, 3] = float(board.has_kingside_castling_rights(chess.BLACK))
        state[:, :, 4] = float(board.has_queenside_castling_rights(chess.BLACK))
        if board.ep_square is not None:
            ep_rank = chess.square_rank(board.ep_square)
            ep_file = chess.square_file(board.ep_square)
            state[7 - ep_rank, ep_file, 5] = 1.0

        return np.concatenate((position, state), axis=-1)
