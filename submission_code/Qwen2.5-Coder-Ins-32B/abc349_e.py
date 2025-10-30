# YOUR CODE HERE
def check_win(board, player):
    # Check rows, columns, and diagonals
    for i in range(3):
        if all(board[i][j] == player for j in range(3)) or all(board[j][i] == player for j in range(3)):
            return True
    if all(board[i][i] == player for i in range(3)) or all(board[i][2 - i] == player for i in range(3)):
        return True
    return False

def minimax(board, depth, is_maximizing, scores):
    if check_win(board, 'R'):
        return scores['R'] - scores['B']
    if check_win(board, 'B'):
        return scores['B'] - scores['R']
    if all(board[i][j] != 'W' for i in range(3) for j in range(3)):
        return scores['R'] - scores['B']

    if is_maximizing:
        max_eval = float('-inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == 'W':
                    board[i][j] = 'R'
                    scores['R'] += board_values[i][j]
                    eval = minimax(board, depth + 1, False, scores)
                    scores['R'] -= board_values[i][j]
                    board[i][j] = 'W'
                    max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == 'W':
                    board[i][j] = 'B'
                    scores['B'] += board_values[i][j]
                    eval = minimax(board, depth + 1, True, scores)
                    scores['B'] -= board_values[i][j]
                    board[i][j] = 'W'
                    min_eval = min(min_eval, eval)
        return min_eval

import sys
input = sys.stdin.read().split()
board_values = [[int(input[i * 3 + j]) for j in range(3)] for i in range(3)]
board = [['W' for _ in range(3)] for _ in range(3)]
scores = {'R': 0, 'B': 0}

result = minimax(board, 0, True, scores)

if result > 0:
    print("Takahashi")
else:
    print("Aoki")