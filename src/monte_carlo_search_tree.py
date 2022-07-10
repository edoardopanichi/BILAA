'''
 *    author:  Ishaan Gupta
 *    created: 9.11.2020 12:04:06       
'''

import chess
import chess.pgn
import chess.engine
import random
import time
from math import log,sqrt,e,inf

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

def expand(curr_node, white):
    
    if(len(curr_node.children) == 0):
        return curr_node
    
    max_ucb = -inf
    if(white):
        idx = -1
        max_ucb = -inf
        sel_child = None
        for i in curr_node.children:
            tmp = ucb1(i)
            if(tmp>max_ucb):
                idx = i
                max_ucb = tmp
                sel_child = i

        return(expand(sel_child, 0))

    else:
        idx = -1
        min_ucb = inf
        sel_child = None
        for i in curr_node.children:
            tmp = ucb1(i)
            if(tmp<min_ucb):
                idx = i
                min_ucb = tmp
                sel_child = i

        return expand(sel_child,1)

def rollback(curr_node,reward):
    curr_node.n+=1
    curr_node.v+=reward
    while(curr_node.parent!=None):
        curr_node.N+=1
        curr_node = curr_node.parent
    return curr_node

def mcts_pred(curr_node, over, white, iterations=10):
    if(over):
        return -1
    all_moves = [curr_node.state.san(i) for i in list(curr_node.state.legal_moves)]
    map_state_move = dict()
    
    for i in all_moves:
        tmp_state = chess.Board(curr_node.state.fen())
        tmp_state.push_san(i)
        child = node()
        child.state = tmp_state
        child.parent = curr_node
        curr_node.children.add(child)
        map_state_move[child] = i
        
    while(iterations>0):
        if(white):
            idx = -1
            max_ucb = -inf
            sel_child = None
            for i in curr_node.children:
                tmp = ucb1(i)
                if(tmp>max_ucb):
                    idx = i
                    max_ucb = tmp
                    sel_child = i
            ex_child = expand(sel_child,0)
            reward,state = rollout(ex_child)
            curr_node = rollback(state,reward)
            iterations-=1
        else:
            idx = -1
            min_ucb = inf
            sel_child = None
            for i in curr_node.children:
                tmp = ucb1(i)
                if(tmp<min_ucb):
                    idx = i
                    min_ucb = tmp
                    sel_child = i

            ex_child = expand(sel_child,1)

            reward,state = rollout(ex_child)

            curr_node = rollback(state,reward)
            iterations-=1
    if(white):
        
        mx = -inf
        idx = -1
        selected_move = ''
        for i in (curr_node.children):
            tmp = ucb1(i)
            if(tmp>mx):
                mx = tmp
                selected_move = map_state_move[i]
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

board = chess.Board()
engine = chess.engine.SimpleEngine.popen_uci(r'C:\Users\ishaa\Desktop\chess_engine\stockfish-11-win\Windows\stockfish_20011801_x64.exe')

white = 1
moves = 0
pgn = []
game = chess.pgn.Game()
evaluations = []
sm = 0
cnt = 0
while((not board.is_game_over())):
    all_moves = [board.san(i) for i in list(board.legal_moves)]
    #start = time.time()
    root = node()
    root.state = board
    result = mcts_pred(root,board.is_game_over(),white)
    #sm+=(time.time()-start)
    board.push_san(result)
    #print(result)
    pgn.append(result)
    white ^= 1
    #cnt+=1
    
    moves+=1
    #board_evaluation = evaluate(board.fen().split()[0])
    #evaluations.append(board_evaluation)
#print("Average Time per move = ",sm/cnt)
print(board)
print(" ".join(pgn))
print()
#print(evaluations)
print(board.result())
game.headers["Result"] = board.result()
#print(game)
engine.quit()