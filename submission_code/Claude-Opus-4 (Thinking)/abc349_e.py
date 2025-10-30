# YOUR CODE HERE
def check_win(board, color):
    # Check rows
    for i in range(3):
        if all(board[i][j] == color for j in range(3)):
            return True
    
    # Check columns  
    for j in range(3):
        if all(board[i][j] == color for i in range(3)):
            return True
    
    # Check diagonals
    if all(board[i][i] == color for i in range(3)):
        return True
    if all(board[i][2-i] == color for i in range(3)):
        return True
    
    return False

def takahashi_wins(board, scores, turn, t_score, a_score, memo):
    # Returns True if Takahashi wins with optimal play from this state
    
    state = (tuple(tuple(row) for row in board), turn, t_score, a_score)
    
    if state in memo:
        return memo[state]
    
    # Find empty cells
    empty = [(i, j) for i in range(3) for j in range(3) if board[i][j] == 0]
    
    if not empty:  # Board full, check scores
        result = t_score > a_score
        memo[state] = result
        return result
    
    if turn == 0:  # Takahashi's turn
        # Takahashi wins if he can find ANY winning move
        for i, j in empty:
            board[i][j] = 1
            new_t_score = t_score + scores[i][j]
            
            # Check if this move wins immediately
            if check_win(board, 1):
                board[i][j] = 0
                memo[state] = True
                return True
            
            # Otherwise, continue game
            if takahashi_wins(board, scores, 1, new_t_score, a_score, memo):
                board[i][j] = 0
                memo[state] = True
                return True
            
            board[i][j] = 0
        
        memo[state] = False
        return False
    
    else:  # Aoki's turn
        # Takahashi wins only if ALL of Aoki's moves lead to Takahashi winning
        for i, j in empty:
            board[i][j] = 2
            new_a_score = a_score + scores[i][j]
            
            # Check if this move wins immediately for Aoki
            if check_win(board, 2):
                board[i][j] = 0
                memo[state] = False
                return False
            
            # Otherwise, continue game
            if not takahashi_wins(board, scores, 0, t_score, new_a_score, memo):
                board[i][j] = 0
                memo[state] = False
                return False
            
            board[i][j] = 0
        
        memo[state] = True
        return True

# Read input
scores = []
for _ in range(3):
    row = list(map(int, input().split()))
    scores.append(row)

# Solve
board = [[0] * 3 for _ in range(3)]
memo = {}

if takahashi_wins(board, scores, 0, 0, 0, memo):
    print("Takahashi")
else:
    print("Aoki")