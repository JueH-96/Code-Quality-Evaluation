def check_winner(grid):
    # Check rows, columns and diagonals for a winner
    for i in range(3):
        if grid[i][0] == grid[i][1] == grid[i][2] and grid[i][0] != 0:
            return grid[i][0]
        if grid[0][i] == grid[1][i] == grid[2][i] and grid[0][i] != 0:
            return grid[0][i]
    if grid[0][0] == grid[1][1] == grid[2][2] and grid[0][0] != 0:
        return grid[0][0]
    if grid[0][2] == grid[1][1] == grid[2][0] and grid[0][2] != 0:
        return grid[0][2]
    return 0

def minimax(grid, depth, is_maximizing, scores):
    winner = check_winner(grid)
    if winner != 0:
        return winner * float('inf')
    if depth == 0:
        return sum(sum(row) for row in scores)

    if is_maximizing:
        best_score = -float('inf')
        for i in range(3):
            for j in range(3):
                if grid[i][j] == 0:
                    grid[i][j] = 1
                    score = minimax(grid, depth - 1, False, scores)
                    grid[i][j] = 0
                    best_score = max(score, best_score)
        return best_score
    else:
        best_score = float('inf')
        for i in range(3):
            for j in range(3):
                if grid[i][j] == 0:
                    grid[i][j] = -1
                    score = minimax(grid, depth - 1, True, scores)
                    grid[i][j] = 0
                    best_score = min(score, best_score)
        return best_score

def solve_game(scores):
    grid = [[0 for _ in range(3)] for _ in range(3)]
    takahashi_score = minimax(grid, 9, True, scores)
    aoki_score = minimax(grid, 9, False, scores)
    return "Takahashi" if takahashi_score > aoki_score else "Aoki"

# Read input
scores = [list(map(int, input().split())) for _ in range(3)]

# Solve the game and print the winner
print(solve_game(scores))