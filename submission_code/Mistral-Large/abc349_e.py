import sys

def read_input():
    grid = []
    for _ in range(3):
        row = list(map(int, input().split()))
        grid.append(row)
    return grid

def check_win(board, color):
    # Check rows, columns, and diagonals for a win
    for i in range(3):
        if all(board[i][j] == color for j in range(3)) or all(board[j][i] == color for j in range(3)):
            return True
    if all(board[i][i] == color for i in range(3)) or all(board[i][2-i] == color for i in range(3)):
        return True
    return False

def evaluate(board, scores):
    if check_win(board, 'R'):
        return (float('inf'), float('-inf'))
    if check_win(board, 'B'):
        return (float('-inf'), float('inf'))
    if all(cell != 'W' for row in board for cell in row):
        if scores[0] > scores[1]:
            return (float('inf'), float('-inf'))
        else:
            return (float('-inf'), float('inf'))
    return (scores[0], scores[1])

def minimax(board, depth, is_maximizing, alpha, beta, scores, grid):
    if check_win(board, 'R') or check_win(board, 'B') or all(cell != 'W' for row in board for cell in row):
        return evaluate(board, scores)

    if is_maximizing:
        max_eval = (float('-inf'), float('inf'))
        for i in range(3):
            for j in range(3):
                if board[i][j] == 'W':
                    board[i][j] = 'R'
                    scores[0] += grid[i][j]
                    eval = minimax(board, depth + 1, False, alpha, beta, scores, grid)
                    board[i][j] = 'W'
                    scores[0] -= grid[i][j]
                    max_eval = max(max_eval, eval, key=lambda x: x[0])
                    alpha = max(alpha, eval[0])
                    if beta <= alpha:
                        break
        return max_eval
    else:
        min_eval = (float('inf'), float('-inf'))
        for i in range(3):
            for j in range(3):
                if board[i][j] == 'W':
                    board[i][j] = 'B'
                    scores[1] += grid[i][j]
                    eval = minimax(board, depth + 1, True, alpha, beta, scores, grid)
                    board[i][j] = 'W'
                    scores[1] -= grid[i][j]
                    min_eval = min(min_eval, eval, key=lambda x: x[1])
                    beta = min(beta, eval[1])
                    if beta <= alpha:
                        break
        return min_eval

def determine_winner(grid):
    board = [['W' for _ in range(3)] for _ in range(3)]
    scores = [0, 0]
    result = minimax(board, 0, True, float('-inf'), float('inf'), scores, grid)
    if result[0] > result[1]:
        return "Takahashi"
    else:
        return "Aoki"

def main():
    grid = read_input()
    winner = determine_winner(grid)
    print(winner)

if __name__ == "__main__":
    main()