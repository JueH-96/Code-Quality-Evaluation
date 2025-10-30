# YOUR CODE HERE
import sys

def spiral_order(n):
    dx = [0, 1, 0, -1]
    dy = [1, 0, -1, 0]
    x, y = 0, 0
    direction = 0
    spiral = [[0] * n for _ in range(n)]
    
    for i in range(1, n*n):
        spiral[x][y] = i
        nx, ny = x + dx[direction], y + dy[direction]
        if 0 <= nx < n and 0 <= ny < n and spiral[nx][ny] == 0:
            x, y = nx, ny
        else:
            direction = (direction + 1) % 4
            x, y = x + dx[direction], y + dy[direction]
    
    return spiral

def print_grid(grid):
    for row in grid:
        print(*row)

n = int(sys.stdin.readline().strip())
grid = spiral_order(n)

center = n // 2
grid[center][center] = 'T'

print_grid(grid)