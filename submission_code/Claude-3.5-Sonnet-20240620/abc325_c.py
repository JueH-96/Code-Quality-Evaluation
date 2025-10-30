# YOUR CODE HERE
import sys

def find_connected_components(grid):
    H, W = len(grid), len(grid[0])
    visited = [[False] * W for _ in range(H)]
    
    def dfs(i, j):
        if i < 0 or i >= H or j < 0 or j >= W or visited[i][j] or grid[i][j] == '.':
            return
        visited[i][j] = True
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                dfs(i + di, j + dj)

    count = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#' and not visited[i][j]:
                dfs(i, j)
                count += 1
    
    return count

H, W = map(int, input().split())
grid = [input().strip() for _ in range(H)]

result = find_connected_components(grid)
print(result)