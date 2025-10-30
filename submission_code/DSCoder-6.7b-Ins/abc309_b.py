# YOUR CODE HERE

import sys

def shift_grid(grid):
    N = len(grid)
    new_grid = [[0]*N for _ in range(N)]

    # Shift outer squares
    for i in range(N):
        if i == 0 or i == N-1:
            new_grid[i] = grid[i][1:] + [grid[0][0]]
        else:
            new_grid[i] = [grid[0][0]] + grid[i][:-1]

    # Shift inner squares
    for i in range(1, N-1):
        for j in range(1, N-1):
            new_grid[i][j] = grid[i-1][j]

    return new_grid

def print_grid(grid):
    for row in grid:
        print(''.join(map(str, row)))

def main():
    N = int(sys.stdin.readline().strip())
    grid = [list(map(int, sys.stdin.readline().strip())) for _ in range(N)]
    new_grid = shift_grid(grid)
    print_grid(new_grid)

if __name__ == "__main__":
    main()