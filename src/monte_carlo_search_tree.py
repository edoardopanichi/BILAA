'''
 *    author:  Ishaan Gupta
 *    created: 9.11.2020 12:04:06  
'''

'''
 *    author: Edoardo Panichi 
 *    modifications started: 10.07.2022 
'''

import chess
import chess.pgn
import chess.engine
import random
import time
from math import log, sqrt, e, inf

class node():
    def __init__(self):
        self.state = chess.Board()
        self.action = ''
        # Children takes track of the branches of the tree 
        self.children = set() # A set is a collection which is unordered, unchangeable, and unindexed.
        # Note: Set items are unchangeable, but you can remove items and add new items.
        self.parent = None
        
        # values for the calculation of the ucb. 
        self.N = 0 # Number of times parent node has been visited
        self.n = 0 # Number of times current node has been visited
        self.v = 0 # Exploitation factor of current node

# This metrics it is used by the MCTS to determine which is the action to take.
# Here 10**-6 and 10**-10 are added to avoid 0 division exception.
def ucb1(curr_node):
    ans = curr_node.v + 2 * (sqrt(log(curr_node.N + e + (10**-6))/(curr_node.n + (10**-10))))
    return ans

def rollout(curr_node):
    
    if(curr_node.state.is_game_over()):
        board = curr_node.state
        if(board.result()=='1-0'):
            #print("h1")
            return (1, curr_node)
        elif(board.result()=='0-1'):
            #print("h2")
            return (-1, curr_node)
        else:
            return (0.5, curr_node)
    
    # given all the legal moves, we generate a list of the move in SAN notation (e.g Be4 - Bishop to e4)
    all_moves = [curr_node.state.san(i) for i in list(curr_node.state.legal_moves)]
    
    for move in all_moves:
        # Forsyth–Edwards Notation (FEN) is a standard notation for describing a particular board position 
        # of a chess game.
        # With the following line of code a new board is created to analyze the evolution of the state through
        # the rollout step
        tmp_state = chess.Board(curr_node.state.fen())
        tmp_state.push_san(move) # play a move
        child = node() 
        child.state = tmp_state
        child.parent = curr_node
        # the newly generated state (where the new move has been played) has to be added to the children list
        # of the current node.
        curr_node.children.add(child)
    
    # In the previous cycle, multiple alternatives lines are played. For each line a possible different legal 
    # move is played. And all the new Boards are saved are children of the current state. Now the algorithm 
    # picks a random children to keep exploring it. That is the definition of ROLLOUT step in a MCTS.   
    rnd_state = random.choice(list(curr_node.children))

    return rollout(rnd_state) # recursive function until the game ends 

# white --> 1
# black --> 0
def expand(curr_node, white):
    
    # when the curr_node has no children we need to find the possible children running ROLLOUT. 
    # The integration of expand and rollout is handled in the mcts_pred function.
    if(len(curr_node.children) == 0):
        return curr_node
    
    max_ucb = -inf
    if(white):
        idx = -1
        max_ucb = -inf
        sel_child = None
        for child in curr_node.children:
            tmp = ucb1(child)
            if(tmp > max_ucb):
                idx = child
                max_ucb = tmp
                sel_child = child

        return(expand(sel_child, 0)) # recursive function until we end up in a state for which the children 
        # have not been defined yet, i.e. we have not executed ROLLOUT phase. Every time we call again the 
        # function expand we change the color, cause the a move has been played.

    else:
        idx = -1
        min_ucb = inf
        sel_child = None
        for child in curr_node.children:
            tmp = ucb1(child)
            if(tmp < min_ucb):
                idx = child
                min_ucb = tmp
                sel_child = child

        return expand(sel_child, 1)

def backpropagation(curr_node, reward):
    curr_node.n += 1
    curr_node.v += reward
    
    # if the current node has a parent, we need to update the value of N for curr_node. Once done that, we 
    # can move one step upwards in the tree to update N of its parent and so on until we reach the starting 
    # node.
    while(curr_node.parent != None):
        curr_node.N += 1
        curr_node = curr_node.parent
        
    return curr_node # at the end of the backpropagation the curr_node will be the starting node, i.e. the 
    # node we started from seeking for the best action to take.

def mcts_pred(curr_node, over, white, iterations = 3):
    
    if(over):
        return -1
    
    # given all the legal moves, we generate a list of the move in SAN notation (e.g Be4 - Bishop to e4)
    all_moves = [curr_node.state.san(i) for i in list(curr_node.state.legal_moves)]
    
    # the following dictionary contains couples of state-move
    map_state_move = dict()
    
    for move in all_moves:
        # With the following line of code a new board is created to analyze the evolution of the state throughout
        # the following steps. A new board is created for each possible move available in curr_node position.
        tmp_state = chess.Board(curr_node.state.fen())
        tmp_state.push_san(move)
        child = node()
        child.state = tmp_state
        child.parent = curr_node
        # the newly generated state (where the new move has been played) has to be added to the children list
        # of the current node.
        curr_node.children.add(child)
        map_state_move[child] = move
        
    while(iterations > 0):
        # We need to select the action that leads to the highest ucb1. Once found that state, we can run over
        # it the expand, rollout and backpropagation functions
        if(white):
            idx = -1
            max_ucb = -inf
            sel_child = None
            for child in curr_node.children:
                tmp = ucb1(child)
                if(tmp > max_ucb):
                    idx = child
                    max_ucb = tmp
                    sel_child = child
                    
            ex_child = expand(sel_child, 0)
            reward, state = rollout(ex_child)
            curr_node = backpropagation(state, reward)
            iterations -= 1
            
        else:
            idx = -1
            min_ucb = inf
            sel_child = None
            for child in curr_node.children:
                tmp = ucb1(child)
                if(tmp < min_ucb):
                    idx = child
                    min_ucb = tmp
                    sel_child = child

            ex_child = expand(sel_child, 1)

            reward, state = rollout(ex_child)

            curr_node = backpropagation(state, reward)
            iterations -= 1
    
    # When the code gets it MCTS has already run for all the required iteration, therefore we have all the 
    # information to infer which action should lead to the best result.       
    if(white):
        mx = -inf
        idx = -1
        selected_move = ''
        
        for child in (curr_node.children):
            tmp = ucb1(child)
            if(tmp > mx):
                mx = tmp
                selected_move = map_state_move[child]
                
        return selected_move
    
    else:
        mn = inf
        idx = -1
        selected_move = ''
        for i in (curr_node.children):
            tmp = ucb1(i)
            if(tmp<mn):
                mn = tmp
                selected_move = map_state_move[i]
        return selected_move

# board = chess.Board()
# #engine = chess.engine.SimpleEngine.popen_uci(r'C:\Users\ishaa\Desktop\chess_engine\stockfish-11-win\Windows\stockfish_20011801_x64.exe')

# white = 1
# moves = 0
# # PGN (Portable Game Notation) is an easy-to-read format which records both the moves of the game 
# # (in standard algebraic notation) and any related data such as the names of the players, the winner/loser, 
# # and even the date the game was played.
# pgn = []
# # To export your game with all headers, comments and variations, you can do it like this:
# game = chess.pgn.Game()

# # evaluations = []
# # sm = 0
# # cnt = 0
# while((not board.is_game_over())):
#     all_moves = [board.san(i) for i in list(board.legal_moves)]
#     root = node()
#     root.state = board
    
#     result = mcts_pred(root, board.is_game_over(), white)
#     board.push_san(result)
#     print("\nmove played:", result)
#     print(board)
    
#     pgn.append(result)
    
#     # the operator ^= Performs Bitwise OR on operands and assign value to left operand. This means that the 
#     # value of white is flipped between 0 and 1 after each move.
#     white ^= 1
#     #cnt+=1
    
#     moves += 1
#     #board_evaluation = evaluate(board.fen().split()[0])
#     #evaluations.append(board_evaluation)
    
# #print("Average Time per move = ",sm/cnt)
# print(board)
# print(" ".join(pgn))
# print()
# #print(evaluations)
# print(board.result())
# game.headers["Result"] = board.result()
# #print(game)
# # engine.quit()