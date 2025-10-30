# YOUR CODE HERE

import sys

def dfs(i, j, grid):
    if i < 0 or j < 0 or i >= len(grid) or j >= len(grid[0]) or grid[i][j] != '#':
        return
    grid[i][j] = '.'
    dfs(i-1, j, grid)
    dfs(i+1, j, grid)
    dfs(i, j-1, grid)
    dfs(i, j+1, grid)

def count_sensors(grid):
    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '#':
                dfs(i, j, grid)
                count += 1
    return count

def main():
    H, W = map(int, sys.stdin.readline().split())
    grid = [list(sys.stdin.readline().strip()) for _ in range(H)]
    print(count_sensors(grid))

if __name__ == "__main__":
    main()