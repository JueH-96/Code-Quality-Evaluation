# YOUR CODE HERE
import sys

# Setting a higher recursion depth limit is generally a good practice for recursive solutions
# although depth 9 might not strictly require it on most systems.
try:
    # Increase recursion depth limit for potentially deep game trees, although max depth is 9.
    # The default limit (often 1000) should be sufficient for depth 9.
    sys.setrecursionlimit(2000) 
except Exception as e: 
    # In some environments (like online judges), changing recursion limit might be restricted.
    # Pass silently if it fails. For depth 9, default limit should suffice.
    pass

# Global dictionary for memoization to store results of computed game states.
# This avoids recomputing the outcome for the same board configuration multiple times.
memo = {}

# Use float('inf') and float('-inf') to represent guaranteed wins by forming a line.
# These values must be larger in magnitude than any possible score difference.
INF = float('inf') 

# Global list to store the grid values flattened into a 1D list (indices 0-8).
# This makes it easy to access the score A[i] for cell index i.
A_flat = []

def check_win(board, player):
    """
    Checks if the given player (1 for Takahashi, 2 for Aoki) has won 
    by forming a line of three of their marks on the board.
    
    Args:
        board: A list representing the current board state (0: empty, 1: Red, 2: Blue).
        player: The player mark (1 or 2) to check for a win.

    Returns:
        True if the player has a winning line, False otherwise.
    """
    
    player_mark = player # Use player value (1 or 2) directly as the mark

    # Check rows for three consecutive marks
    for i in range(3):
        if board[i*3] == player_mark and board[i*3+1] == player_mark and board[i*3+2] == player_mark:
            return True
            
    # Check columns for three consecutive marks
    for i in range(3):
        if board[i] == player_mark and board[i+3] == player_mark and board[i+6] == player_mark:
            return True
            
    # Check main diagonal (top-left to bottom-right)
    if board[0] == player_mark and board[4] == player_mark and board[8] == player_mark:
        return True
        
    # Check anti-diagonal (top-right to bottom-left)
    if board[2] == player_mark and board[4] == player_mark and board[6] == player_mark:
        return True
        
    # If no winning line is found after checking all possibilities
    return False

def minimax(board_tuple):
    """
    Recursive minimax function with memoization to determine the outcome of the game.
    It calculates the optimal outcome from the perspective of Takahashi (the maximizer), 
    assuming both players play optimally.

    Args:
        board_tuple: A tuple representing the current state of the 3x3 grid. 
                     Using a tuple allows it to be used as a dictionary key for memoization.
                     Cell values: 0 for empty, 1 for Takahashi (Red), 2 for Aoki (Blue).

    Returns:
        The evaluation score of the state:
        - INF if Takahashi can force a win by forming a line from this state.
        - -INF if Aoki can force a win by forming a line from this state.
        - The final score difference (Takahashi's score - Aoki's score) if the game ends 
          by filling the board without a line win.
    """
    
    # Use the board state tuple itself as the key for memoization lookup.
    state_key = board_tuple 
    if state_key in memo:
        # If this state has been evaluated before, return the stored result.
        return memo[state_key]

    # Convert the board tuple to a list to allow modifications during move exploration.
    board = list(board_tuple) 

    # --- Base Cases for Recursion ---

    # Base Case 1: Check if a player has already won by forming a line.
    # This checks if the player who made the *last* move (to reach this state) won.
    if check_win(board, 1): # Takahashi wins by line
        memo[state_key] = INF
        return INF
    if check_win(board, 2): # Aoki wins by line
        memo[state_key] = -INF
        return -INF

    # Find all available (empty) cells where the next move can be made.
    available_moves = [i for i, cell in enumerate(board) if cell == 0]
    
    # Base Case 2: Check if the board is full (no available moves left).
    if not available_moves: 
        # Game ends, calculate final scores based on occupied cells.
        score_T = 0 # Takahashi's total score
        score_A = 0 # Aoki's total score
        for i in range(9):
            if board[i] == 1: # Cell taken by Takahashi
                score_T += A_flat[i]
            elif board[i] == 2: # Cell taken by Aoki
                score_A += A_flat[i]
        
        # Return the score difference. The problem guarantees the total sum is odd,
        # so Takahashi's score (S_T) cannot equal Aoki's score (S_A).
        result = score_T - score_A 
        memo[state_key] = result
        return result

    # --- Recursive Step ---

    # Determine whose turn it is based on the number of pieces already placed.
    # Takahashi (player 1) is Red, Aoki (player 2) is Blue.
    num_red = board.count(1)
    num_blue = board.count(2)
    
    # If the counts are equal, it's Takahashi's turn (Maximizer).
    # If Takahashi has placed one more piece, it's Aoki's turn (Minimizer).
    current_player_is_Takahashi = (num_red == num_blue)

    if current_player_is_Takahashi: # Takahashi's turn (Maximizer)
        # Initialize with the worst possible outcome for the maximizer.
        max_eval = -INF 
        # Explore each available move.
        for i in available_moves:
            board[i] = 1 # Try placing Takahashi's mark (1) at cell i.
            # Recursively call minimax for the resulting board state (now it's Aoki's turn).
            eval_res = minimax(tuple(board)) 
            # Update the maximum evaluation found so far for Takahashi.
            max_eval = max(max_eval, eval_res)
            # Backtrack: undo the move by resetting the cell to empty (0).
            board[i] = 0 
        
        # Store the computed optimal value for this state in the memo table.
        memo[state_key] = max_eval
        # Return the best outcome Takahashi can achieve from this state.
        return max_eval

    else: # Aoki's turn (Minimizer), which occurs when num_red == num_blue + 1
        # Initialize with the worst possible outcome for the minimizer (best for Takahashi).
        min_eval = INF 
        # Explore each available move.
        for i in available_moves:
            board[i] = 2 # Try placing Aoki's mark (2) at cell i.
            # Recursively call minimax for the resulting board state (now it's Takahashi's turn).
            eval_res = minimax(tuple(board)) 
            # Update the minimum evaluation found so far (best for Aoki, worst for Takahashi).
            min_eval = min(min_eval, eval_res)
            # Backtrack: undo the move.
            board[i] = 0 
            
        # Store the computed optimal value for this state in the memo table.
        memo[state_key] = min_eval
        # Return the best outcome Aoki can achieve (minimum score difference for Takahashi).
        return min_eval

# --- Main execution part of the script ---
if __name__ == '__main__':
    # Read the 3x3 grid values from standard input.
    # Flatten the grid into a 1D list `A_flat` for easy access using cell index (0-8).
    A_flat = []
    for _ in range(3):
        A_flat.extend(list(map(int, sys.stdin.readline().split())))

    # The initial state of the game is an empty board, represented by a tuple of 9 zeros.
    initial_board_tuple = tuple([0] * 9)
    
    # Start the minimax search from the initial empty board state. 
    # The function returns the evaluated outcome assuming optimal play.
    final_value = minimax(initial_board_tuple)

    # Determine the winner based on the sign of the final evaluation value.
    # A positive value means Takahashi wins (either by score S_T > S_A or by forming a line, represented as INF).
    # A negative value means Aoki wins (either by score S_A > S_T or by forming a line, represented as -INF).
    # The value cannot be zero because the total sum of scores is guaranteed to be odd.
    if final_value > 0: 
        print("Takahashi")
    else: 
        print("Aoki")