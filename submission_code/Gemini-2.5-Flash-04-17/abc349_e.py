import sys

# Increase recursion depth for minimax search
# Maximum depth is 9, but internal function calls add overhead. 3000 is a safe buffer.
sys.setrecursionlimit(3000)

# Read input grid A
A = []
for _ in range(3):
    A.append(list(map(int, sys.stdin.readline().split())))

# Check for 3-in-a-row win
# Returns 1 for red (T), 2 for blue (A), 0 for no win
def check_win(board):
    # Check rows
    for i in range(3):
        if board[i][0] != 0 and board[i][0] == board[i][1] and board[i][1] == board[i][2]:
            return board[i][0]
    # Check columns
    for j in range(3):
        if board[0][j] != 0 and board[0][j] == board[1][j] and board[1][j] == board[2][j]:
            return board[0][j]
    # Check diagonals
    if board[0][0] != 0 and board[0][0] == board[1][1] and board[1][1] == board[2][2]:
        return board[0][0]
    if board[0][2] != 0 and board[0][2] == board[1][1] and board[1][1] == board[2][0]:
        return board[0][2]
    return 0 # No win

# Count pieces on the board
def count_pieces(board):
    count = 0
    for r in board:
        for cell in r:
            if cell != 0:
                count += 1
    return count

# Memoization for game_outcome (checks if 3-in-a-row is forced)
memo_outcome = {}

# Determine game outcome based on 3-in-a-row using minimax with alpha-beta pruning
# Returns 1 if Takahashi (red) can force a win, -1 if Aoki (blue) can force a win,
# 0 if neither can force a win/loss by 3-in-a-row (game goes to score).
# alpha: best value Takahashi can guarantee so far (initially -inf)
# beta: best value Aoki can guarantee so far (initially +inf)
def game_outcome_ab(board, alpha, beta):
    k = count_pieces(board)
    winner = check_win(board)

    # Base cases: terminal states
    if winner == 1: return 1 # Red (T) wins
    if winner == 2: return -1 # Blue (A) wins
    if k == 9: return 0 # Board full, no 3-in-a-row win

    # Memoization check
    board_tuple = tuple(tuple(row) for row in board)
    if board_tuple in memo_outcome:
        return memo_outcome[board_tuple]

    # k is even: Takahashi's turn (maximizing player for {1, 0, -1})
    if k % 2 == 0:
        max_res = -2 # Value lower than any possible outcome (-1)
        for i in range(3):
            for j in range(3):
                if board[i][j] == 0: # If cell is white
                    new_board = [list(row) for row in board] # Create a copy
                    new_board[i][j] = 1 # Place red
                    
                    res = game_outcome_ab(new_board, alpha, beta)
                    max_res = max(max_res, res)
                    
                    # Alpha-beta pruning
                    alpha = max(alpha, max_res)
                    if beta <= alpha:
                        break # Beta cut-off
            if beta <= alpha:
                break # Beta cut-off

        memo_outcome[board_tuple] = max_res
        return max_res

    # k is odd: Aoki's turn (minimizing player for {1, 0, -1})
    else:
        min_res = 2 # Value higher than any possible outcome (1)
        for i in range(3):
            for j in range(3):
                if board[i][j] == 0: # If cell is white
                    new_board = [list(row) for row in board] # Create a copy
                    new_board[i][j] = 2 # Place blue
                    
                    res = game_outcome_ab(new_board, alpha, beta)
                    min_res = min(min_res, res)
                    
                    # Alpha-beta pruning
                    beta = min(beta, min_res)
                    if beta <= alpha:
                        break # Alpha cut-off
            if beta <= alpha:
                break # Alpha cut-off

        memo_outcome[board_tuple] = min_res
        return min_res

# Memoization for solve_score_values (calculates score difference)
memo_score_values = {}

# Calculate the optimal additional score difference (T - A) from this state until k=9
# This function is called only if game_outcome determined the game goes to score.
# alpha: best value Takahashi can guarantee so far (initially -inf)
# beta: best value Aoki can guarantee so far (initially +inf)
def solve_score_values_ab(board, k, alpha, beta):
    # Base case: board is full
    # When k=9 is reached, there are no more moves, so the additional score difference is 0.
    if k == 9:
        return 0

    # Memoization check
    board_tuple = tuple(tuple(row) for row in board)
    if board_tuple in memo_score_values:
        return memo_score_values[board_tuple]

    # k is even: Takahashi's turn (maximizing player for additional score difference)
    if k % 2 == 0:
        max_additional_score_diff = -float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == 0: # If cell is white
                    new_board = [list(row) for row in board] # Create a copy
                    new_board[i][j] = 1 # Place red
                    
                    # Score contribution from this move: A[i][j] goes to T, increasing T-A difference by A[i][j].
                    current_move_contribution = A[i][j]
                    
                    # Recursive call for the next state (Aoki's turn)
                    # This returns the optimal additional score difference from k+1 to k=9.
                    additional_score_from_next_state = solve_score_values_ab(new_board, k + 1, alpha, beta)
                    
                    total_additional_score_from_here = current_move_contribution + additional_score_from_next_state
                    
                    max_additional_score_diff = max(max_additional_score_diff, total_additional_score_from_here)

                    # Alpha-beta pruning
                    alpha = max(alpha, max_additional_score_diff)
                    if beta <= alpha:
                         break # Beta cut-off
            if beta <= alpha:
                 break # Beta cut-off


        memo_score_values[board_tuple] = max_additional_score_diff
        return max_additional_score_diff

    # k is odd: Aoki's turn (minimizing player for additional score difference)
    else:
        min_additional_score_diff = float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == 0: # If cell is white
                    new_board = [list(row) for row in board] # Create a copy
                    new_board[i][j] = 2 # Place blue
                    
                    # Score contribution from this move: A[i][j] goes to A, decreasing T-A difference by A[i][j].
                    current_move_contribution = -A[i][j]
                    
                    # Recursive call for the next state (Takahashi's turn)
                    # This returns the optimal additional score difference from k+1 to k=9.
                    additional_score_from_next_state = solve_score_values_ab(new_board, k + 1, alpha, beta)
                    
                    total_additional_score_from_here = current_move_contribution + additional_score_from_next_state
                    
                    min_additional_score_diff = min(min_additional_score_diff, total_additional_score_from_here)

                    # Alpha-beta pruning
                    beta = min(beta, min_additional_score_diff)
                    if beta <= alpha:
                         break # Alpha cut-off
            if beta <= alpha:
                 break # Alpha cut-off

        memo_score_values[board_tuple] = min_additional_score_diff
        return min_additional_score_diff

# Initial state
initial_board = [[0] * 3 for _ in range(3)]
initial_k = 0

# Step 1: Determine if a player can force a 3-in-a-row win/loss using minimax with alpha-beta pruning
# Initial call with alpha = -inf, beta = +inf
outcome_win_loss = game_outcome_ab(initial_board, -float('inf'), float('inf'))

if outcome_win_loss == 1:
    print("Takahashi")
elif outcome_win_loss == -1:
    print("Aoki")
else: # outcome_win_loss == 0, game goes to score
    # Step 2: Calculate the final score difference assuming game goes to k=9
    # The initial score difference at k=0 is 0.
    # The total final score difference = initial_score_diff + solve_score_values_ab(initial_board, initial_k, -inf, +inf).
    final_score_diff = 0 + solve_score_values_ab(initial_board, initial_k, -float('inf'), float('inf'))
    
    if final_score_diff > 0:
        print("Takahashi")
    else: # final_score_diff < 0 because total sum is odd, so diff is non-zero if k=9
        print("Aoki")