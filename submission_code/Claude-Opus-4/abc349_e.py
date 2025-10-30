# YOUR CODE HERE
def check_winner(board):
    # Check rows
    for i in range(3):
        if board[i][0] != 0 and board[i][0] == board[i][1] == board[i][2]:
            return board[i][0]
    
    # Check columns
    for j in range(3):
        if board[0][j] != 0 and board[0][j] == board[1][j] == board[2][j]:
            return board[0][j]
    
    # Check diagonals
    if board[0][0] != 0 and board[0][0] == board[1][1] == board[2][2]:
        return board[0][0]
    if board[0][2] != 0 and board[0][2] == board[1][1] == board[2][0]:
        return board[0][2]
    
    return 0

def minimax(grid, board, takahashi_score, aoki_score, is_takahashi_turn, memo):
    # Create state key for memoization
    state = tuple(tuple(row) for row in board)
    key = (state, takahashi_score, aoki_score, is_takahashi_turn)
    
    if key in memo:
        return memo[key]
    
    # Check if someone won
    winner = check_winner(board)
    if winner == 1:  # Takahashi won
        memo[key] = True
        return True
    elif winner == -1:  # Aoki won
        memo[key] = False
        return False
    
    # Check if board is full
    empty_cells = [(i, j) for i in range(3) for j in range(3) if board[i][j] == 0]
    if not empty_cells:
        # Game ends, higher score wins
        memo[key] = takahashi_score > aoki_score
        return takahashi_score > aoki_score
    
    if is_takahashi_turn:
        # Takahashi wants to maximize his winning chances
        for i, j in empty_cells:
            board[i][j] = 1  # Takahashi's move
            new_takahashi_score = takahashi_score + grid[i][j]
            if minimax(grid, board, new_takahashi_score, aoki_score, False, memo):
                board[i][j] = 0  # Undo move
                memo[key] = True
                return True
            board[i][j] = 0  # Undo move
        memo[key] = False
        return False
    else:
        # Aoki wants to minimize Takahashi's winning chances
        for i, j in empty_cells:
            board[i][j] = -1  # Aoki's move
            new_aoki_score = aoki_score + grid[i][j]
            if not minimax(grid, board, takahashi_score, new_aoki_score, True, memo):
                board[i][j] = 0  # Undo move
                memo[key] = False
                return False
            board[i][j] = 0  # Undo move
        memo[key] = True
        return True

# Read input
grid = []
for _ in range(3):
    row = list(map(int, input().split()))
    grid.append(row)

# Initialize board (0 = empty, 1 = Takahashi, -1 = Aoki)
board = [[0] * 3 for _ in range(3)]

# Run minimax with memoization
memo = {}
takahashi_wins = minimax(grid, board, 0, 0, True, memo)

if takahashi_wins:
    print("Takahashi")
else:
    print("Aoki")