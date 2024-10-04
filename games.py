"""Games or Adversarial Search (Chapter 5)"""
# Author: Wen Luo (Thomas) Lui 
# Student Id: 301310026

import copy
import itertools
import random
from collections import namedtuple

import numpy as np

from utils import vector_add

GameState = namedtuple('GameState', 'to_move, utility, board, moves')


def gen_state(to_move='X', x_positions=[], o_positions=[], h=3, v=3):
    """Given whose turn it is to move, the positions of X's on the board, the
    positions of O's on the board, and, (optionally) number of rows, columns
    and how many consecutive X's or O's required to win, return the corresponding
    game state"""

    moves = set([(x, y) for x in range(1, h + 1) for y in range(1, v + 1)]) - set(x_positions) - set(o_positions)
    moves = list(moves)
    board = {}
    for pos in x_positions:
        board[pos] = 'X'
    for pos in o_positions:
        board[pos] = 'O'
    return GameState(to_move=to_move, utility=0, board=board, moves=moves)

    

# ______________________________________________________________________________
# MinMax Search


def minmax(game, state):
    """Given a state in a game, calculate the best move by searching
    forward all the way to the terminal states. [Figure 5.3]"""

    player = game.to_move(state)

    def max_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a)))
        return v

    def min_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a)))
        return v

    # Body of minmax_decision:
    return max(game.actions(state), key=lambda a: min_value(game.result(state, a)))

# *********
# A bug with this assignment is that the game doesn't display when the 
# player loses the game but it will disable the game to prevent further player movement.
# **********************


# The minmax_cutoff function calculate the best move by searching
# forward up to the specified cutoff depth. At the cutoff depth, evaluate the positions
# using the provided evaluation function. This function represents the max node in the
# minimax algorithm with depth cutoff, where the player aims to maximize the expected value
# of the chosen action up to the cutoff depth. The algorithm then returns the best action
# based on the evaluated values.

# Works up to 6 by 6 board size with depth set to 2

def minmax_cutoff(game, state):
    """Given a state in a game, calculate the best move by searching
    forward all the way to the cutoff depth. At that level use evaluation func."""


    boardCutoffDepth = game.d

    if game.h == 3:
        boardCutoffDepth = 1
    if game.h >= 6:
        boardCutoffDepth = 2
   
    boardWidth = game.h
    boardHeight = game.v
    currentDepth = 0
    # print("the board size width within minmax_cutoff: ", boardWidth)
    # print("the board size height within minmax_cutoff: ", boardHeight)
    
    player = game.to_move(state)
    
    # Max node in the minimax algorithm with depth cutoff. Represents the player's decision node
    # where the player chooses the action that maximizes the expected value up to the cutoff depth.
    def max_value(state, currentDepth):
        if game.terminal_test(state) or currentDepth >= boardCutoffDepth:
            # using the evaluation function to calculate the best score when the current depth is greater the user set cutoff depth value
            return game.evaluation_func(state)
            
        v = -np.inf
        for a in game.actions(state):
            # Evaluate the value of each possible action using the 'min_value' function.
            v = max(v, min_value(game.result(state, a), currentDepth+1))
        return v

    def min_value(state, currentDepth):
        if game.terminal_test(state) or currentDepth >= boardCutoffDepth:
            # using the evaluation function to calculate the best score when the current depth is greater the user set cutoff depth value
            return game.evaluation_func(state)
            
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a), currentDepth+1))
        return v
    
    
    return max(game.actions(state), key=lambda a: min_value(game.result(state, a), currentDepth))

    

# ______________________________________________________________________________


#  The best result for expect_minmax is to have the cutoff depth value set to 3 as it will block off potential winning position of the player based on testing
#  This algorithm implementation is based on the lecture slide pseudocode which ignores the min_value function as the chance_node function replaces it as min
#  The game board size can work above 6 by 6 for this implementation of expectimax.

def expect_minmax(game, state):
    """
    [Figure 5.11]
    Return the best move for a player after dice are thrown. The game tree
	includes chance nodes along with min and max nodes.
	"""
    currentDepth = 0
    # boardCutoffDepth = game.d
    boardCutoffDepth = 3
    player = game.to_move(state)


    #  Max node in the expectiminimax algorithm. It represents the player's decision node where the player chooses the action that maximizes the expected value.
    def max_value(state,currentDepth):
        
        # if game.terminal_test(state) or currentDepth >= boardCutoffDepth:
        #     return game.utility(state, player)
        
        v = -np.inf
        
        for a in game.actions(state):
            
            # v = max(v, chance_node(state, a, currentDepth + 1))
            v = max(v, value(game.result(state, a), a, currentDepth + 1))
        return v
    
    # not used as chance node is the exp-value function for expectimax
    def min_value(state):
        v = np.inf
        for a in game.actions(state):
            v = min(v, chance_node(state, a))
        return v
    
    # value function implemented with the lecture slides for check terminal state or depth cutoff
    # it also uses the next agent to determine which max or chance node function to call for the node value 
    def value(state, action, currentDepth):
        result_state = game.result(state, action)
        if game.terminal_test(state) or currentDepth >= boardCutoffDepth:
            return game.utility(result_state, player)
        
        # determines whether the next agent is the player or opponent and call max_value or chance_node accordingly
        if game.to_move == player: 
            return max_value(result_state, currentDepth + 1)
        else:
            return chance_node(result_state, action, currentDepth+1)
        
        
    #  Chance node in the expectiminimax algorithm. It represents a situation where
    #  uncertainty arises, and the expected value is calculated based on possible outcomes.
    def chance_node(state, action, currentDepth):
        
        result_state = game.result(state, action)
                
        sum_chances = 0
        num_chances = len(game.chances(result_state))
        # print("the num chance value is: ", num_chances)
        
        if num_chances == 0:
            # setting it to a default probability outcome of rolling two dice 1/36 chances
            total_prob = 1.0/36
        else:
            # used the number of all possible states from the current result state and calculate a probability for this chance node
            total_prob = 1.0/num_chances 
        
        
        for a in game.actions(result_state):
        
            sum_chances = sum_chances + total_prob * value(game.result(result_state, a), a , currentDepth + 1)
            
        return sum_chances
    

    # Body of expect_minmax:
    return max(game.actions(state), key=lambda a: value(state, a, currentDepth), default=None)


# This alpha beta search with cutoff function employs the alpha-beta pruning algorithm to efficiently explore the game tree
# and determine the best action for the player at the current state. The search goes up to a
# specified depth or until terminal states are reached. The utility values are evaluated using
# the provided evaluation function.
#      This algorithm works best with depth = 2 and can work on board size greater than 6.

def alpha_beta_search(game, state):
    """Search game to determine best action; use alpha-beta pruning.
    As in [Figure 5.7], this version searches all the way to the leaves."""

    currentDepth = 0
    boardCutoffDepth = game.d #possiblility set it to default depth = 2
    # print("the board cuttoff at alpha is: ", boardCutoffDepth)

    if boardCutoffDepth >= 2:
        boardCutoffDepth = 2
    if game.h == 3:
        boardCutoffDepth = -1


    player = game.to_move(state)

    # Functions used by alpha_beta
    def max_value(state, alpha, beta, currentDepth):
        if game.terminal_test(state) or currentDepth >= boardCutoffDepth:
            # Terminal state or cutoff depth reached, return evaluation.
            # return game.utility(state, player)
            return game.evaluation_func(state)
        v = -np.inf
        for a in game.actions(state):
            # Explore the possible actions and update the maximum value.
            v = max(v, min_value(game.result(state, a), alpha, beta, currentDepth + 1))
            if v >= beta:
                # Prune the search if the value is greater than or equal to beta.
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta, currentDepth):
        if game.terminal_test(state) or currentDepth >= boardCutoffDepth:
            # Terminal state or cutoff depth reached, return evaluation.
            # return game.utility(state, player)
            return game.evaluation_func(state)
        
        v = np.inf
        for a in game.actions(state):
            # Explore the possible actions and update the minimum value.
            v = min(v, max_value(game.result(state, a), alpha, beta, currentDepth + 1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    # Body of alpha_beta_search:
    best_action = None
    best_val = -np.inf
    alpha = -np.inf
    beta = np.inf
    for action in game.actions(state):
        # Iterate through possible actions and update the best action and value.
        val = min_value(game.result(state, action), alpha, beta, currentDepth)
        if val > best_val:
            best_val = val
            best_action = action
        alpha = max(alpha, best_val)

    return best_action


def alpha_beta_cutoff_search(game, state, d=4, cutoff_test=None, eval_fn=None):
    """Search game to determine best action; use alpha-beta pruning.
    This version cuts off search and uses an evaluation function."""
    print("alpha_beta_cutoff_search: may be used, if so, must be implemented by students")
    
    return None


# ______________________________________________________________________________
# Players for Games


def query_player(game, state):
    """Make a move by querying standard input."""
    print("current state:")
    game.display(state)
    print("available moves: {}".format(game.actions(state)))
    print("")
    move = None
    if game.actions(state):
        move_string = input('Your move? ')
        try:
            move = eval(move_string)
        except NameError:
            move = move_string
    else:
        print('no legal moves: passing turn to next player')
    return move


def random_player(game, state):
    """A player that chooses a legal move at random."""
    return random.choice(game.actions(state)) if game.actions(state) else None


def alpha_beta_player(game, state):
    return alpha_beta_search(game, state)


def minmax_player(game,state):
    if( game.d == -1):
        return minmax(game, state)
    return minmax_cutoff(game, state)


def expect_minmax_player(game, state):
    return expect_minmax(game, state)


# ______________________________________________________________________________
# 


class Game:
    """A game is similar to a problem, but it has a utility for each
    state and a terminal test instead of a path cost and a goal
    test. To create a game, subclass this class and implement actions,
    result, utility, and terminal_test. You may override display and
    successors or you can inherit their default methods. You will also
    need to set the .initial attribute to the initial state; this can
    be done in the constructor."""

    def actions(self, state):
        """Return a list of the allowable moves at this point."""
        raise NotImplementedError

    def result(self, state, move):
        """Return the state that results from making a move from a state."""
        raise NotImplementedError

    def utility(self, state, player):
        """Return the value of this final state to player."""
        raise NotImplementedError

    def terminal_test(self, state):
        """Return True if this is a final state for the game."""
        return not self.actions(state)

    def to_move(self, state):
        """Return the player whose move it is in this state."""
        return state.to_move

    def display(self, state):
        """Print or otherwise display the state."""
        print(state)

    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)

    def play_game(self, *players):
        """Play an n-person, move-alternating game."""
        state = self.initial
        while True:
            for player in players:
                move = player(self, state)
                state = self.result(state, move)
                if self.terminal_test(state):
                    self.display(state)
                    return self.utility(state, self.to_move(self.initial))



class TicTacToe(Game):
    """Play TicTacToe on an h x v board, with Max (first player) playing 'X'.
    A state has the player to_move, a cached utility, a list of moves in
    the form of a list of (x, y) positions, and a board, in the form of
    a dict of {(x, y): Player} entries, where Player is 'X' or 'O'.
    depth = -1 means max search tree depth to be used."""

    def __init__(self, h=3, v=3, k=3, d=-1):
        self.h = h
        self.v = v
        self.k = k
        self.depth = d
        moves = [(x, y) for x in range(1, h + 1)
                 for y in range(1, v + 1)]
        self.initial = GameState(to_move='X', utility=0, board={}, moves=moves)

    def actions(self, state):
        """Legal moves are any square not yet taken."""
        return state.moves

    def result(self, state, move):
        if move not in state.moves:
            return state  # Illegal move has no effect
        board = state.board.copy()
        board[move] = state.to_move
        moves = list(state.moves)
        moves.remove(move)
        return GameState(to_move=('O' if state.to_move == 'X' else 'X'),
                         utility=self.compute_utility(board, move, state.to_move),
                         board=board, moves=moves)

    def utility(self, state, player):
        """Return the value to player; 1 for win, -1 for loss, 0 otherwise."""
        return state.utility if player == 'X' else -state.utility

    def terminal_test(self, state):
        """A state is terminal if it is won or there are no empty squares."""
        return state.utility != 0 or len(state.moves) == 0

    def display(self, state):
        board = state.board
        for x in range(1, self.h + 1):
            for y in range(1, self.v + 1):
                print(board.get((x, y), '.'), end=' ')
            print()

    def compute_utility(self, board, move, player):
        """If 'X' wins with this move, return 1; if 'O' wins return -1; else return 0."""
        if (self.k_in_row(board, move, player, (0, 1)) or
                self.k_in_row(board, move, player, (1, 0)) or
                self.k_in_row(board, move, player, (1, -1)) or
                self.k_in_row(board, move, player, (1, 1))):
            # return self.k if player == 'X' else -self.k
            return +1 if player == 'X' else -1
        else:
            return 0


    # Computes the heuristic value for a player's position on the board after a move.
    # The heuristic evaluates the state based on the difference between values calculated for 'X' and 'O',
    # considering rows, columns, diagonals, and anti-diagonals. It aims to guide the player towards favorable positions.
    # Returns the heuristic value, where higher values favor the current player and lower values favor the opponent.
    def evaluation_func(self, state):
        """
        Computes the value for a player on the board after a move.
        Returns the difference between values calculated for 'X' and 'O'.
        """

        player = state.to_move
        opponent = 'O'
        player = 'O' if opponent == 'X' else 'X'

        
        def evaluate_line(line, player):
            """
            Helper function to evaluate a line (row, column, diagonal, or anti-diagonal).
            """
            player_count = line.count(player)
            opponent_count = line.count('X' if player == 'O' else 'O')


            if player_count == 0 and opponent_count == 0:
                return 0  # Empty line
            elif player_count > 0 and opponent_count == 0:
                return 10 ** player_count  # Favorable for the current player
            elif player_count == 0 and opponent_count > 0:
                return -10 ** opponent_count  # Favorable for the opponent
            elif player_count == self.h:
                return 100000  # if player win with a line
            elif opponent_count == self.h:
                return -100000  # if opponent win with a line
            else:
                return 0  # Mixed line
            

        
        board = state.board
        width = self.h
        height = self.v

        # print("the utility is: ", state.utility)
        if state.utility == self.k:
            return np.inf if player == 'X' else -np.inf
        elif state.utility == -self.k:
            return -np.inf if player == 'X' else np.inf


        
        # Evaluate lines for 'X' and 'O'
        # x_total_value = evaluate_lines(board, width, height, player)
        row_values_x = [evaluate_line([board.get((x, y), '.') for y in range(1, height + 1)], player) for x in range(1, width + 1)]
        col_values_x = [evaluate_line([board.get((x, y), '.') for x in range(1, width + 1)], player) for y in range(1, height + 1)]
        x_total_value = sum(row_values_x) + sum(col_values_x)

        # o_total_value = evaluate_lines(board, width, height, opponent)
        row_values_o = [evaluate_line([board.get((x, y), '.') for y in range(1, height + 1)], opponent) for x in range(1, width + 1)]
        col_values_o = [evaluate_line([board.get((x, y), '.') for x in range(1, width + 1)], opponent) for y in range(1, height + 1)]
        o_total_value = sum(row_values_o) + sum(col_values_o)

        # Evaluate main diagonal for 'X' and 'O'
        x_main_diag_value = evaluate_line([board.get((i, i), '.') for i in range(1, min(width, height) + 1)], player)
        o_main_diag_value = evaluate_line([board.get((i, i), '.') for i in range(1, min(width, height) + 1)], opponent)


        # Evaluate anti-diagonal for 'X' and 'O'
        x_anti_diag_value = evaluate_line([board.get((i, height - i + 1), '.') for i in range(1, min(width, height) + 1)], player)
        o_anti_diag_value = evaluate_line([board.get((i, height - i + 1), '.') for i in range(1, min(width, height) + 1)], opponent)


        # Sum up the values for 'X' and 'O'
        x_total_value += x_main_diag_value + x_anti_diag_value
        o_total_value += o_main_diag_value + o_anti_diag_value



        return x_total_value - o_total_value


		
    def k_in_row(self, board, move, player, delta_x_y):
        """Return true if there is a line through move on board for player.
        hint: This function can be extended to test of n number of items on a line 
        not just self.k items as it is now. """
        (delta_x, delta_y) = delta_x_y
        x, y = move
        n = 0  # n is number of moves in row
        while board.get((x, y)) == player:
            n += 1
            x, y = x + delta_x, y + delta_y
        x, y = move
        while board.get((x, y)) == player:
            n += 1
            x, y = x - delta_x, y - delta_y
        n -= 1  # Because we counted move itself twice
        return n >= self.k


    def chances(self, state):
        """Return a list of all possible states."""
        
        chance_states = []

        if not self.terminal_test(state):
            chance_actions = self.actions(state)
            
            # gets the list of possible action and apply those actions to get the resulting state and append it to the chance_state
            for chance_action in chance_actions:
                chance_result = self.result(state, chance_action)
                chance_states.append(chance_result)

        return chance_states
        

        
    
class Gomoku(TicTacToe):
    """Also known as Five in a row."""

    def __init__(self, h=15, v=16, k=5):
        TicTacToe.__init__(self, h, v, k)








