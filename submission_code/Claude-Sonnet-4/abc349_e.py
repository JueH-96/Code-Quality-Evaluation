def check_winner(board):
    # Check rows, columns, and diagonals for 3 in a row
    # 1 = Takahashi (red), -1 = Aoki (blue), 0 = white
    
    # Check rows
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != 0:
            return board[i][0]
    
    # Check columns
    for j in range(3):
        if board[0][j] == board[1][j] == board[2][j] != 0:
            return board[0][j]
    
    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] != 0:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != 0:
        return board[0][2]
    
    return 0  # No winner yet

def is_board_full(board):
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:
                return False
    return True

def minimax(board, values, is_takahashi_turn, takahashi_score, aoki_score, alpha, beta):
    # Check if game is over
    winner = check_winner(board)
    if winner == 1:  # Takahashi wins by 3 in a row
        return 1
    elif winner == -1:  # Aoki wins by 3 in a row
        return -1
    elif is_board_full(board):  # Game ends, check scores
        if takahashi_score > aoki_score:
            return 1  # Takahashi wins
        else:
            return -1  # Aoki wins (can't tie since sum is odd)
    
    if is_takahashi_turn:
        # Takahashi wants to maximize
        max_eval = -2
        for i in range(3):
            for j in range(3):
                if board[i][j] == 0:  # Empty cell
                    board[i][j] = 1  # Takahashi's move
                    new_takahashi_score = takahashi_score + values[i][j]
                    eval_score = minimax(board, values, False, new_takahashi_score, aoki_score, alpha, beta)
                    board[i][j] = 0  # Undo move
                    max_eval = max(max_eval, eval_score)
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break
            if beta <= alpha:
                break
        return max_eval
    else:
        # Aoki wants to minimize
        min_eval = 2
        for i in range(3):
            for j in range(3):
                if board[i][j] == 0:  # Empty cell
                    board[i][j] = -1  # Aoki's move
                    new_aoki_score = aoki_score + values[i][j]
                    eval_score = minimax(board, values, True, takahashi_score, new_aoki_score, alpha, beta)
                    board[i][j] = 0  # Undo move
                    min_eval = min(min_eval, eval_score)
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break
            if beta <= alpha:
                break
        return min_eval

# Read input
values = []
for _ in range(3):
    row = list(map(int, input().split()))
    values.append(row)

# Initialize empty board (0 = white, 1 = red/Takahashi, -1 = blue/Aoki)
board = [[0 for _ in range(3)] for _ in range(3)]

# Run minimax to determine winner
result = minimax(board, values, True, 0, 0, -2, 2)

if result == 1:
    print("Takahashi")
else:
    print("Aoki")