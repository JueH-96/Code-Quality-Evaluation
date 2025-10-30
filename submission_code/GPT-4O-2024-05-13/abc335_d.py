# YOUR CODE HERE
def generate_grid(N):
    grid = [[0] * N for _ in range(N)]
    center = N // 2
    grid[center][center] = 'T'
    
    num = 1
    x, y = 0, 0
    dx, dy = 0, 1
    
    for _ in range(N * N):
        if grid[x][y] == 0:
            grid[x][y] = num
            num += 1
        if x + dx < 0 or x + dx >= N or y + dy < 0 or y + dy >= N or grid[x + dx][y + dy] != 0:
            dx, dy = dy, -dx
        x += dx
        y += dy
    
    grid[center][center] = 'T'
    return grid

def print_grid(grid):
    for row in grid:
        print(" ".join(map(str, row)))

import sys
input = sys.stdin.read
N = int(input().strip())

grid = generate_grid(N)
print_grid(grid)