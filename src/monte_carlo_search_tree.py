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
from math import log, sqrt, e, inf
import numpy as np
from evaluation_class import evaluation

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
        
        

class MCTS:
    def __init__(self):
        # Instance of the evaluation function to have access to its functions
        self.eval = evaluation()

    # This metrics it is used by the MCTS to determine which is the action to take.
    # Here 10**-6 and 10**-10 are added to avoid 0 division exception.
    def ucb1(self, curr_node):
        ans = curr_node.v + 2 * (sqrt(log(curr_node.N + e + (10**-6))/(curr_node.n + (10**-10))))
        return ans

    def rollout(self, curr_node):
        
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

        return self.rollout(rnd_state) # recursive function until the game ends 

    # white --> 1
    # black --> 0
    def expand(self, curr_node, white):
        
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
                tmp = self.ucb1(child)
                if(tmp > max_ucb):
                    idx = child
                    max_ucb = tmp
                    sel_child = child

            return(self.expand(sel_child, 0)) # recursive function until we end up in a state for which the children 
            # have not been defined yet, i.e. we have not executed ROLLOUT phase. Every time we call again the 
            # function expand we change the color, cause the a move has been played.

        else:
            idx = -1
            min_ucb = inf
            sel_child = None
            for child in curr_node.children:
                tmp = self.ucb1(child)
                if(tmp < min_ucb):
                    idx = child
                    min_ucb = tmp
                    sel_child = child

            return self.expand(sel_child, 1)

    def backpropagation(self, curr_node, reward):
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

    def mcts_pred(self, curr_node, over, white, iterations = 3):
        
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
                    tmp = self.ucb1(child)
                    if(tmp > max_ucb):
                        idx = child
                        max_ucb = tmp
                        sel_child = child
                        
                ex_child = self.expand(sel_child, 0)
                reward, state = self.rollout(ex_child)
                curr_node = self.backpropagation(state, reward)
                iterations -= 1
                
            else:
                idx = -1
                min_ucb = inf
                sel_child = None
                for child in curr_node.children:
                    tmp = self.ucb1(child)
                    if(tmp < min_ucb):
                        idx = child
                        min_ucb = tmp
                        sel_child = child

                ex_child = self.expand(sel_child, 1)

                reward, state = self.rollout(ex_child)

                curr_node = self.backpropagation(state, reward)
                iterations -= 1
        
        # When the code gets it MCTS has already run for all the required iteration, therefore we have all the 
        # information to infer which action should lead to the best result.       
        if(white):
            mx = -inf
            idx = -1
            selected_move = ''
            
            for child in (curr_node.children):
                tmp = self.ucb1(child)
                if(tmp > mx):
                    mx = tmp
                    selected_move = map_state_move[child]
                    
            return selected_move
        
        else:
            mn = inf
            idx = -1
            selected_move = ''
            for i in (curr_node.children):
                tmp = self.ucb1(i)
                if(tmp<mn):
                    mn = tmp
                    selected_move = map_state_move[i]
            return selected_move

    # The following algorithm implements a simpler version of the MCST where each leaf of the tree is explored 
    # for 5 steps and not till the end of the game. This algorithm relies on a NN to evaluate a given position.
    def simple_mcst(self, board, evaluation_score, epochs = 5, depth = 5):

        first_legal_moves = list(board.legal_moves)
        # initialization of the scores to one for each available legal move.
        scores = np.ones(len(first_legal_moves))
        
        # According to the number of epochs and depth we randomly explore the possible actions given the 
        # initial board position.
        for epoch in range(epochs):

            for first_move in range(len(first_legal_moves)):
                # Copying the board set-up to try a new line of the game.
                play_board = board.copy()
                # Playing a move among the legal ones
                play_board.push(first_legal_moves[first_move])

                # each legal move of the starting position we play n (=depth) extra random moves to understand
                # to which position we might end-up.
                for _ in range(depth):
                    legal_moves = list(play_board.legal_moves)
                    # Doing an if over a list gives: 
                    # List is empty --> False, 
                    # List is not empty --> True
                    if legal_moves:
                        move = random.choice(legal_moves)
                        play_board.push(move)
                    else:
                        break
                
                # If we want to evaluate a given position with the NN we need to translate the data into 
                # 8x8x12 inputs.   
                translated = np.array(self.eval.translate(play_board))
                scores[first_move] += evaluation_score(translated)
        # We pick the move that leads to the state with the highest score.        
        idx = np.where(scores == max(scores))[0][0]
        
        return first_legal_moves[idx], scores
        