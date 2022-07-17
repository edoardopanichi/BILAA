from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense,Flatten

# Simple NN model to evaluate a position on the board. The input shape is 8x8x12 because a chess game uses a 
# 8x8 board and there are 12 different possible pieces 8: 6 whites and 6 blacks (King, Queen, Rook, Bishop, 
# Knight, Pawn)
class evaluator:
    def __init__(self):
        
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
    def simple_eval_model(self, image_shape = (8, 8, 12)):
        
        model = Sequential()
        model.add(Dense(10, input_shape = image_shape))
        model.add(Dense(10, activation = 'relu'))
        model.add(Flatten())
        model.add(Dense(1))
        
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
        return foo