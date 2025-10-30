import sys

def solve(n):
    # Create an empty grid
    grid = [[0]*n for _ in range(n)]

    # Place Takahashi at the center
    grid[n//2][n//2] = 'T'

    # Initialize variables
    x, y = n//2, n//2
    dx, dy = [0, 0, 1, -1], [1, -1, 0, 0]
    dir = 0

    # Place dragon parts
    for i in range(1, n*n):
        nx, ny = x + dx[dir], y + dy[dir]
        if 0 <= nx < n and 0 <= ny < n and grid[nx][ny] == 0:
            grid[nx][ny] = i
            x, y = nx, ny
        else:
            dir = (dir + 1) % 4
            nx, ny = x + dx[dir], y + dy[dir]
            grid[nx][ny] = i
            x, y = nx, ny

    # Print the grid
    for row in grid:
        print(' '.join(map(str, row)))

# Read input
n = int(sys.stdin.readline())

# Solve and print the answer
solve(n)