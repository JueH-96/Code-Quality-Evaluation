import numpy as np

def check_win(grid, player):
    for i in range(3):
        if np.all(grid[i, :] == player) or np.all(grid[:, i] == player):
            return True
    if np.all(np.diag(grid) == player) or np.all(np.diag(np.fliplr(grid)) == player):
        return True
    return False

def score(grid):
    return np.sum(grid * np.where(grid > 0, 1, -1))

def game(grid, player):
    if check_win(grid, player):
        return True
    if np.all(grid != 0):
        return score(grid) > 0
    for i in range(3):
        for j in range(3):
            if grid[i, j] == 0:
                grid[i, j] = player
                if not game(grid, -player):
                    grid[i, j] = 0
                    return True
                grid[i, j] = 0
    return False

A = np.array([list(map(int, input().split())) for _ in range(3)])
total = np.sum(A)
if total > 0:
    A = np.where(A > 0, 1, -1)
else:
    A = np.where(A < 0, -1, 1)

if game(A, 1):
    print("Takahashi")
else:
    print("Aoki")