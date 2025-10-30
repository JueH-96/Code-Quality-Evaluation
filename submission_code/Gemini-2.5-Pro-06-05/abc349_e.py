import sys

# Set a higher recursion limit for safety, though 9 levels is shallow.
sys.setrecursionlimit(2 * 10**5)

# Globals for the score grid and the memoization table.
# Using globals simplifies passing them through the recursion.
A = []
memo = {}

# Constants to represent players and empty cells for better readability.
WHITE = 0
TAKAHASHI = 1  # Player 1, will paint cells red.
AOKI = 2       # Player 2, will paint cells blue.

def check_win(grid, player):
    """
    Checks if the specified player has formed a line of three of their color.
    A line can be a row, a column, or a diagonal.
    """
    # Check rows
    for r in range(3):
        if all(grid[r][c] == player for c in range(3)):
            return True
    # Check columns
    for c in range(3):
        if all(grid[r][c] == player for r in range(3)):
            return True
    # Check diagonals
    if grid[0][0] == player and grid[1][1] == player and grid[2][2] == player:
        return True
    if grid[0][2] == player and grid[1][1] == player and grid[2][0] == player:
        return True
    return False

def get_scores(grid):
    """
    Calculates the total scores for both players based on the cells they've claimed.
    """
    t_score, a_score = 0, 0
    for r in range(3):
        for c in range(3):
            if grid[r][c] == TAKAHASHI:
                t_score += A[r][c]
            elif grid[r][c] == AOKI:
                a_score += A[r][c]
    return t_score, a_score

def find_winner(grid):
    """
    Recursively determines the winner from the current grid state using the minimax algorithm.
    It assumes optimal play from both sides.
    The function returns the constant representing the winning player (TAKAHASHI or AOKI).
    Memoization is used to store results for previously seen grid states to avoid recomputation.
    """
    # Create a hashable representation of the grid for the memoization key.
    grid_tuple = tuple(map(tuple, grid))
    if grid_tuple in memo:
        return memo[grid_tuple]

    # Count the number of moves made to determine the turn and check for the end of the game.
    moves_made = 9 - sum(row.count(WHITE) for row in grid)

    # Base Case 1: The grid is full. The game ends, and the winner is decided by score.
    if moves_made == 9:
        t_score, a_score = get_scores(grid)
        # The sum of all scores is guaranteed to be odd, so no draws are possible.
        result = TAKAHASHI if t_score > a_score else AOKI
        memo[grid_tuple] = result
        return result

    # Determine whose turn it is. Takahashi goes first (moves 0, 2, 4, ...).
    if moves_made % 2 == 0:  # Takahashi's turn
        current_player = TAKAHASHI
        opponent = AOKI
    else:  # Aoki's turn
        current_player = AOKI
        opponent = TAKAHASHI

    # Recursive Step: The current player will try all available moves.
    # They will win if they can find at least one move that leads to a winning state for them.
    for r in range(3):
        for c in range(3):
            if grid[r][c] == WHITE:  # If the cell is available
                # Create a new grid state for the potential move.
                next_grid = [row[:] for row in grid]
                next_grid[r][c] = current_player
                
                # Base Case 2: The move results in an immediate line win.
                if check_win(next_grid, current_player):
                    memo[grid_tuple] = current_player
                    return current_player
                
                # Recurse to find the winner from the next state.
                # If the sub-problem results in a win for the current player,
                # they have found an optimal move and will take it.
                if find_winner(next_grid) == current_player:
                    memo[grid_tuple] = current_player
                    return current_player
    
    # If the loop completes, it means every possible move for the current player
    # leads to a state where the opponent wins. Therefore, the current player loses.
    memo[grid_tuple] = opponent
    return opponent

def solve():
    """
    Main function to read input, run the solver, and print the output.
    """
    global A
    # Read the 3x3 score grid from standard input.
    try:
        A = [list(map(int, sys.stdin.readline().split())) for _ in range(3)]
    except (IOError, ValueError):
        # Handle potential empty input during testing or EOF
        return

    # Initialize the game with an empty 3x3 grid.
    initial_grid = [[WHITE for _ in range(3)] for _ in range(3)]
    
    # Determine the winner assuming optimal play from the initial state.
    winner = find_winner(initial_grid)
    
    # Print the name of the winner.
    if winner == TAKAHASHI:
        print("Takahashi")
    else:
        print("Aoki")

solve()