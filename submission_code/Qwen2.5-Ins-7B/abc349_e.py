# YOUR CODE HERE
import sys

def check_win(board, player):
    for i in range(3):
        if all(board[i][j] == player for j in range(3)) or all(board[j][i] == player for j in range(3)):
            return True
    if all(board[i][i] == player for i in range(3)) or all(board[i][2-i] == player for i in range(3)):
        return True
    return False

def solve(board):
    def dfs(board, turn):
        if check_win(board, 'T'):
            return 1
        if check_win(board, 'A'):
            return -1
        if not any('.' in row for row in board):
            return 0
        
        for i in range(3):
            for j in range(3):
                if board[i][j] == '.':
                    board[i][j] = 'T' if turn else 'A'
                    score = dfs(board, not turn)
                    board[i][j] = '.'
                    if turn and score == 1 or not turn and score == -1:
                        return score
        return 0
    
    return 'Takahashi' if dfs(board, True) == 1 else 'Aoki'

input_board = []
for _ in range(3):
    input_board.append(list(sys.stdin.readline().strip().split()))

print(solve(input_board))