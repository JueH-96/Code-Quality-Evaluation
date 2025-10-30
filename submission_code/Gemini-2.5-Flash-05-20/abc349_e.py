import sys

# Global variables for the A_ij values and memoization table
A = []
memo = {}

# Predefined lines for win checking (rows, columns, diagonals)
# Each tuple contains (row_index, col_index) for cells in a line
LINES = [
    # Rows
    [(0, 0), (0, 1), (0, 2)],
    [(1, 0), (1, 1), (1, 2)],
    [(2, 0), (2, 1), (2, 2)],
    # Columns
    [(0, 0), (1, 0), (2, 0)],
    [(0, 1), (1, 1), (2, 1)],
    [(0, 2), (1, 2), (2, 2)],
    # Diagonals
    [(0, 0), (1, 1), (2, 2)],
    [(0, 2), (1, 1), (2, 0)]
]

# Helper function to check for a win condition (3 in a row)
# Returns 1 if Takahashi (Red) wins, 2 if Aoki (Blue) wins, 0 otherwise
def check_win(grid):
    for line in LINES:
        r1, c1 = line[0]
        r2, c2 = line[1]
        r3, c3 = line[2]

        cell1 = grid[r1][c1]
        cell2 = grid[r2][c2]
        cell3 = grid[r3][c3]

        if cell1 != 0 and cell1 == cell2 and cell2 == cell3:
            return cell1 # 1 for Red (Takahashi), 2 for Blue (Aoki)
    return 0

# Helper function to calculate Takahashi's score
def calculate_t_score(grid):
    score = 0
    for r in range(3):
        for c in range(3):
            if grid[r][c] == 1: # Red cell
                score += A[r][c]
    return score

# Helper function to calculate Aoki's score
def calculate_a_score(grid):
    score = 0
    for r in range(3):
        for c in range(3):
            if grid[r][c] == 2: # Blue cell
                score += A[r][c]
    return score

# Helper function to count filled cells (depth of the game tree)
def calculate_depth(grid):
    count = 0
    for r in range(3):
        for c in range(3):
            if grid[r][c] != 0:
                count += 1
    return count

# Minimax function with Alpha-Beta Pruning
# Returns:
#   float('inf') if Takahashi wins by 3-in-a-row
#   float('-inf') if Aoki wins by 3-in-a-row
#   1 if Takahashi wins by score (after full board, no 3-in-a-row)
#   -1 if Aoki wins by score (after full board, no 3-in-a-row)
def minimax(grid, alpha, beta):
    # Convert list of lists grid to tuple of tuples for memoization key
    grid_tuple = tuple(tuple(row) for row in grid)

    # Check memoization table
    if grid_tuple in memo:
        return memo[grid_tuple]

    # Calculate current game state attributes
    depth = calculate_depth(grid)
    turn = depth % 2 # 0 for Takahashi (Red, Maximizing), 1 for Aoki (Blue, Minimizing)

    # Base case: Check for immediate win (3-in-a-row)
    winner = check_win(grid)
    if winner == 1:
        return float('inf') # Takahashi wins
    if winner == 2:
        return float('-inf') # Aoki wins

    # Base case: Full board (no white cells left)
    if depth == 9:
        t_score = calculate_t_score(grid)
        a_score = calculate_a_score(grid)
        if t_score > a_score:
            return 1 # Takahashi wins by score
        else: # a_score > t_score since sum is odd, scores won't be equal
            return -1 # Aoki wins by score

    # Recursive step
    if turn == 0: # Takahashi's turn (Maximizing player)
        max_eval = -float('inf')
        for r in range(3):
            for c in range(3):
                if grid[r][c] == 0: # If cell is white
                    # Make a move
                    new_grid = [list(row) for row in grid] # Create a mutable copy
                    new_grid[r][c] = 1 # Takahashi paints red
                    
                    eval = minimax(new_grid, alpha, beta)
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    
                    # Alpha-beta pruning
                    if beta <= alpha:
                        break # Prune this branch
            else: # This `else` block belongs to the inner `for` loop
                continue # If `break` was not executed, continue to outer loop's next iteration
            break # If `break` was executed in inner loop, break outer loop too (to stop iterating over moves)
        
        memo[grid_tuple] = max_eval
        return max_eval

    else: # Aoki's turn (Minimizing player)
        min_eval = float('inf')
        for r in range(3):
            for c in range(3):
                if grid[r][c] == 0: # If cell is white
                    # Make a move
                    new_grid = [list(row) for row in grid] # Create a mutable copy
                    new_grid[r][c] = 2 # Aoki paints blue
                    
                    eval = minimax(new_grid, alpha, beta)
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    
                    # Alpha-beta pruning
                    if beta <= alpha:
                        break # Prune this branch
            else: # This `else` block belongs to the inner `for` loop
                continue # If `break` was not executed, continue to outer loop's next iteration
            break # If `break` was executed in inner loop, break outer loop too (to stop iterating over moves)
        
        memo[grid_tuple] = min_eval
        return min_eval

# Main execution block
if __name__ == "__main__":
    # Read input A_ij matrix
    for _ in range(3):
        A.append(list(map(int, sys.stdin.readline().split())))

    # Initialize the grid with all white cells (0)
    initial_grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    # Start the minimax search
    # Initial alpha is -inf, initial beta is +inf
    result = minimax(initial_grid, -float('inf'), float('inf'))

    # Determine the winner based on the result
    if result > 0: # Takahashi wins (either by 3-in-a-row or by score)
        print("Takahashi")
    else: # Aoki wins (either by 3-in-a-row or by score)
        print("Aoki")