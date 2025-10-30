def check_winner(board):
    # Check rows and columns
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] and board[i][0] != 0:
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] and board[0][i] != 0:
            return board[0][i]
    
    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != 0:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != 0:
        return board[0][2]
    
    return None

def minimax(board, turn, score_takahashi, score_aoki):
    winner = check_winner(board)
    if winner == 1:  # Takahashi wins
        return score_takahashi + 1
    elif winner == 2:  # Aoki wins
        return score_aoki + 1
    elif all(cell != 0 for row in board for cell in row):  # No white cells left
        return score_takahashi if score_takahashi > score_aoki else score_aoki
    
    if turn == 1:  # Takahashi's turn
        best_score = float('-inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == 0:  # White cell
                    board[i][j] = 1  # Takahashi's color
                    score = minimax(board, 2, score_takahashi + board[i][j], score_aoki)
                    best_score = max(best_score, score)
                    board[i][j] = 0  # Undo move
        return best_score
    else:  # Aoki's turn
        best_score = float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == 0:  # White cell
                    board[i][j] = 2  # Aoki's color
                    score = minimax(board, 1, score_takahashi, score_aoki + board[i][j])
                    best_score = min(best_score, score)
                    board[i][j] = 0  # Undo move
        return best_score

def main():
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    board = [[0] * 3 for _ in range(3)]
    
    for i in range(3):
        for j in range(3):
            board[i][j] = int(data[i * 3 + j])
    
    score_takahashi = minimax(board, 1, 0, 0)
    
    if score_takahashi > 0:
        print("Takahashi")
    else:
        print("Aoki")

if __name__ == "__main__":
    main()