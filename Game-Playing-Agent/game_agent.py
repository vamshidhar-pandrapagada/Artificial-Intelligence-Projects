"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import itertools
import numpy as np


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """This function outputs the score which is equal to the difference between the number of Player's moves
    and the 2 times number of opponent's remaining moves.This score chases the opponent aggresively.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    player_moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(abs(player_moves - 2*opponent_moves))


def custom_score_2(game, player):
    """This function evaluates the game progress and if the board is occupied at near 70%,
    then check the move state of the player and opponent for its presence in any of the corners.
    If in the corner and board occupancy is  > 70 penalize the move by deducting higher number of points.
    If in the corner and board occupancy is  < 70 reward the move by adding lower number of points.
    A weighted linear difference of (product of score and moves left) for player and opponent is returned. 

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    corners = [(0,0), (0,game.width-1), (game.height-1,0), (game.height-1,game.width-1)]
    remaining_spaces = game.get_blank_spaces()
    game_progress_percentage =  int((len(remaining_spaces)/(game.width * game.height)) * 100)
    
    
    player_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    player_score = 0
    opponent_score = 0
    #player_moves_left = 0
    #opponent_moves_left = 0
    
    for move in player_moves:
        if (move in corners) and game_progress_percentage < 70:
            player_score += 20
        elif (move in corners) and game_progress_percentage > 70:
            player_score -= 50
        #else:
           #player_moves_left += 8 # Total moves left are 15 if board is occupied at 70%

    for move in opponent_moves:
        if (move in corners) and game_progress_percentage < 70:
            opponent_score += 20
        elif (move in corners) and game_progress_percentage > 70:
            opponent_score -= 50
        #else:
         #  opponent_moves_left += 8

    #return float(player_score*player_moves_left) - float(2*opponent_score * opponent_moves_left)   
    #return float((player_score*len(player_moves)) - (2*opponent_score * len(opponent_moves)))   
    return float(player_score - 2*opponent_score) + float (len(player_moves) - 2*len(opponent_moves))


def custom_score_3(game, player):
    """This function evaluates and returns the maximum squared distance between the player and any of the walls.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    walls = [
        [(0, i) for i in range(game.width)],
        [(i, 0) for i in range(game.height)],
        [(game.width - 1, i) for i in range(game.width)],
        [(i, game.height - 1) for i in range(game.height)]
    ]
    
    player_loc = game.get_player_location(player)
    if player_loc == None:
        return 0.
    merged = list(itertools.chain(*walls))
    distance_from_walls = (player_loc - np.array(merged))**2
    abs_distance = ([pair[0]+ pair[1] for pair in distance_from_walls])
    
    
    return float(max(abs_distance))


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth = 3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)
        
        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            #return self.minimax(game, self.search_depth)
            return self.minimax(game, self.search_depth)
            
        except SearchTimeout:
            pass # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax_helper(self, game, depth, maximizing_player = True):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()        
       
        #If No legal Moves,return the utility for the active player on the board
        #This is the terminal test
        if not legal_moves:
            if maximizing_player == True:
                return float("-inf"), (-1, -1)
            else:
                return float("inf"), (-1, -1)
        
        maximizing_score = float("-inf")
        minimizing_score = float("inf")
        best_move = (-1,-1)

        if depth == 1:
            if maximizing_player == True:
                for move in legal_moves:
                    score = self.score(game.forecast_move(move), self)
                    if score > maximizing_score:
                        maximizing_score, best_move = score, move
                return maximizing_score, best_move
            else:
                for move in legal_moves:
                    score = self.score(game.forecast_move(move), self)
                    if score < minimizing_score:
                        minimizing_score, best_move = score, move
                return minimizing_score, best_move
    
        if maximizing_player:
            # calculate the highest score for maximizing player
            for move in legal_moves:
                # Switch the active player by forecsting the move.
                next_board_state = game.forecast_move(move)
                score, temporary_move = self.minimax_helper(next_board_state, depth - 1, False)
                if score > maximizing_score:
                    maximizing_score, best_move = score, move
            return maximizing_score, best_move
        else: #  If minimizing player
           # calculate the minimum score for minimizing player
           for move in legal_moves:
                next_board_state = game.forecast_move(move)
                score, temporary_move = self.minimax_helper(next_board_state, depth - 1, True)
                if score < minimizing_score:
                    minimizing_score, best_move = score, move
           return minimizing_score, best_move  
        
    def minimax(self, game, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        score, best_move = self.minimax_helper(game, depth)
        return best_move


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        self.time_left = time_left
        best_move = (-1, -1)
        
        try:
            # Search method using alpha-beta is built into the try block 
            # to avoid the timeout
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            for level in itertools.count():
                best_move =  self.alphabeta(game, depth = level + 1)
        except SearchTimeout:
             pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def alphabeta_helper(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player = True):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        
        legal_moves = game.get_legal_moves()        
       
        #If No legal Moves,return the utility for the active player on the board
        #This is the terminal test
        if not legal_moves:
            if maximizing_player == True:
                return float("-inf"), (-1, -1)
            else:
                return float("inf"), (-1, -1)

        maximizing_score = float("-inf")
        minimizing_score = float("inf")
        best_move = (-1,-1)
        
        if depth == 1:
            if maximizing_player == True:
                for move in legal_moves:
                   score = self.score(game.forecast_move(move), self)
                   if score > maximizing_score:
                        maximizing_score, best_move = score, move
                   if maximizing_score >= beta:
                        return maximizing_score, best_move
                return maximizing_score, best_move
            else:
                for move in legal_moves:
                    score = self.score(game.forecast_move(move), self)
                    if score < minimizing_score:
                        minimizing_score, best_move = score, move
                    if minimizing_score <= alpha:
                        return minimizing_score, best_move
                return minimizing_score, best_move
        
              
        if maximizing_player:
            # calculate the highest score for maximizing player
            for move in legal_moves:
                # Switch the active player by forecsting the move.
                next_board_state = game.forecast_move(move)
                score, temporary_move = self.alphabeta_helper(next_board_state, depth - 1, alpha, beta, False)
                if score > maximizing_score:
                    maximizing_score, best_move = score, move
                 # Alpha-Beta Pruning
                if maximizing_score >= beta:
                    return maximizing_score, best_move
                # Update alpha, if necessary
                alpha = max(alpha, maximizing_score)
            return maximizing_score, best_move
        else: #  If minimizing player
           # calculate the minimum score for minimizing player
            for move in legal_moves:
                next_board_state = game.forecast_move(move)
                score, temporary_move = self.alphabeta_helper(next_board_state, depth - 1,alpha, beta, True)
                if score < minimizing_score:
                    minimizing_score, best_move = score, move
                # Alpha-Beta Pruning
                if minimizing_score <= alpha:
                    return minimizing_score, best_move
                # Update beta, if necessary
                beta = min(beta, minimizing_score)
            return minimizing_score, best_move  
        
    def alphabeta(self, game, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        score, best_move = self.alphabeta_helper(game, depth)
        return best_move