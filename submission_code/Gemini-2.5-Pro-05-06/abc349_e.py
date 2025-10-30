import sys

# Global dictionary for memoization
memo = {}
# Global 2D list for scores A_i,j
A_scores_grid = []

# Precompute winning line configurations (flat indices 0-8 for a 3x3 grid)
WINNING_LINES_FLAT = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
    [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
    [0, 4, 8], [2, 4, 6]              # Diagonals
]

# Helper function to calculate scores for Takahashi (sT) and Aoki (sA)
# board_config is a flat tuple of 9 cell states (0: white, 1: red/Takahashi, 2: blue/Aoki)
def get_scores(board_config):
    sT = 0
    sA = 0
    for i in range(9):
        r, c = i // 3, i % 3 # Convert flat index to 2D grid coordinates
        if board_config[i] == 1:  # Red cell (Takahashi)
            sT += A_scores_grid[r][c]
        elif board_config[i] == 2:  # Blue cell (Aoki)
            sA += A_scores_grid[r][c]
    return sT, sA

# Helper function to check if a line has been formed
# Returns 1 if red wins, 2 if blue wins, 0 otherwise
def check_line_winner(board_config):
    for line in WINNING_LINES_FLAT:
        cell1_val = board_config[line[0]]
        cell2_val = board_config[line[1]]
        cell3_val = board_config[line[2]]
        
        if cell1_val != 0 and cell1_val == cell2_val and cell1_val == cell3_val:
            return cell1_val  # Return the color (1 for red, 2 for blue)
    return 0 # No line winner

# Minimax function
# board_config: current board state (tuple of 9 ints)
# turn_count: number of moves made so far (0 to 9)
# Returns 1 if Takahashi wins from this state, -1 if Aoki wins.
def minimax(board_config, turn_count):
    memo_key = (board_config, turn_count)
    if memo_key in memo:
        return memo[memo_key]

    current_player_id = turn_count % 2 # 0 for Takahashi, 1 for Aoki
    
    # Base Case: Board is full (9 moves made).
    # This is reached if the 9th move (by Takahashi) did not result in a line.
    # Game ends by score comparison.
    if turn_count == 9:
        sT, sA = get_scores(board_config)
        if sT > sA:
            result = 1  # Takahashi wins
        else: # sA > sT (guaranteed no tie by problem statement)
            result = -1 # Aoki wins
        memo[memo_key] = result
        return result

    # Make a mutable copy of the board for simulating moves
    board_list_mutable = list(board_config)

    if current_player_id == 0:  # Takahashi's turn (Maximizer)
        max_eval = -float('inf') # Takahashi wants to achieve outcome 1
        
        for i in range(9): # Iterate through all cells
            if board_list_mutable[i] == 0:  # If cell is white (available)
                board_list_mutable[i] = 1  # Takahashi paints it red (color 1)
                new_board_config_tuple = tuple(board_list_mutable)

                # Check if Takahashi wins by forming a line with this move
                line_winner_code = check_line_winner(new_board_config_tuple)
                if line_winner_code == 1:  # Takahashi forms a red line
                    current_eval = 1
                else:
                    # No line formed by Takahashi, game continues to Aoki's turn
                    current_eval = minimax(new_board_config_tuple, turn_count + 1)
                
                max_eval = max(max_eval, current_eval)
                board_list_mutable[i] = 0  # Backtrack: unpaint cell for next iteration path
                
                # Optimization: If Takahashi can force a win (outcome 1), no need to explore further.
                if max_eval == 1:
                    break
        
        memo[memo_key] = max_eval
        return max_eval

    else:  # Aoki's turn (Minimizer)
        min_eval = float('inf') # Aoki wants to achieve outcome -1
        
        for i in range(9): # Iterate through all cells
            if board_list_mutable[i] == 0:  # If cell is white (available)
                board_list_mutable[i] = 2  # Aoki paints it blue (color 2)
                new_board_config_tuple = tuple(board_list_mutable)

                # Check if Aoki wins by forming a line with this move
                line_winner_code = check_line_winner(new_board_config_tuple)
                if line_winner_code == 2:  # Aoki forms a blue line
                    current_eval = -1
                else:
                    # No line formed by Aoki, game continues to Takahashi's turn
                    current_eval = minimax(new_board_config_tuple, turn_count + 1)

                min_eval = min(min_eval, current_eval)
                board_list_mutable[i] = 0  # Backtrack
                
                # Optimization: If Aoki can force a win (outcome -1), no need to explore further.
                if min_eval == -1:
                    break
        
        memo[memo_key] = min_eval
        return min_eval

def main():
    global A_scores_grid # Allow modification of the global variable
    A_scores_grid = []
    for _ in range(3):
        A_scores_grid.append(list(map(int, sys.stdin.readline().split())))

    # Initial state: empty board (all 0s), 0 moves made (Takahashi's turn)
    initial_board_config = tuple([0] * 9) 
    
    final_outcome = minimax(initial_board_config, 0)

    if final_outcome == 1:
        print("Takahashi")
    else: # final_outcome == -1
        print("Aoki")

if __name__ == '__main__':
    main()