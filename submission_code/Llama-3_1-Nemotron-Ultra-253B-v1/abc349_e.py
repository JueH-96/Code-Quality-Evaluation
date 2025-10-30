import sys
from functools import lru_cache

A = [list(map(int, sys.stdin.readline().split())) for _ in range(3)]

def has_three_in_a_row(grid, color):
    for i in range(3):
        if all(grid[i][j] == color for j in range(3)):
            return True
    for j in range(3):
        if all(grid[i][j] == color for i in range(3)):
            return True
    if all(grid[i][i] == color for i in range(3)):
        return True
    if all(grid[i][2 - i] == color for i in range(3)):
        return True
    return False

@lru_cache(maxsize=None)
def can_win(grid, turn):
    current_color = 'R' if turn == 0 else 'B'
    # Check for immediate three-in-a-row
    for i in range(3):
        for j in range(3):
            if grid[i][j] == 'W':
                new_grid = list(map(list, grid))
                new_grid[i][j] = current_color
                new_grid = tuple(map(tuple, new_grid))
                if has_three_in_a_row(new_grid, current_color):
                    return True
    # Explore all possible moves
    for i in range(3):
        for j in range(3):
            if grid[i][j] == 'W':
                new_grid = list(map(list, grid))
                new_grid[i][j] = current_color
                new_grid = tuple(map(tuple, new_grid))
                # Check if the new grid is full
                if 'W' not in [cell for row in new_grid for cell in row]:
                    T = sum(A[x][y] for x in range(3) for y in range(3) if new_grid[x][y] == 'R')
                    A_sum = sum(A[x][y] for x in range(3) for y in range(3) if new_grid[x][y] == 'B')
                    if (turn == 0 and T > A_sum) or (turn == 1 and A_sum > T):
                        return True
                else:
                    if not can_win(new_grid, 1 - turn):
                        return True
    return False

initial_grid = (('W', 'W', 'W'), ('W', 'W', 'W'), ('W', 'W', 'W'))
print("Takahashi" if can_win(initial_grid, 0) else "Aoki")