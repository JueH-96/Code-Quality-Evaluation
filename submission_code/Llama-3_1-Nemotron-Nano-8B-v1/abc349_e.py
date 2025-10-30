import sys
from functools import lru_cache

def main():
    grid = [list(map(int, sys.stdin.readline().split())) for _ in range(3)]

    @lru_cache(maxsize=None)
    def dfs(current_grid_tuple, turn, score_t, score_a):
        current_grid = [list(row) for row in current_grid_tuple]

        if has_three_in_a_row(current_grid, 'R' if turn == 0 else 'B'):
            return turn == 0

        if all(cell != 'W' for row in current_grid for cell in row):
            return score_t > score_a

        for i in range(3):
            for j in range(3):
                if current_grid[i][j] == 'W':
                    new_grid = [row.copy() for row in current_grid]
                    new_grid[i][j] = 'R' if turn == 0 else 'B'
                    new_grid_tuple = tuple(tuple(row) for row in new_grid)
                    new_score_t = score_t + (grid[i][j] if turn == 0 else 0)
                    new_score_a = score_a + (grid[i][j] if turn == 1 else 0)
                    if not dfs(new_grid_tuple, 1 - turn, new_score_t, new_score_a):
                        return True
        return False

    initial_grid_tuple = tuple(tuple(['W'] * 3) for _ in range(3))
    result = dfs(initial_grid_tuple, 0, 0, 0)
    print("Takahashi" if result else "Aoki")

def has_three_in_a_row(grid, player):
    for row in grid:
        if all(cell == player for cell in row):
            return True
    for col in range(3):
        if all(grid[row][col] == player for row in range(3)):
            return True
    if all(grid[i][i] == player for i in range(3)):
        return True
    if all(grid[i][2-i] == player for i in range(3)):
        return True
    return False

if __name__ == "__main__":
    main()