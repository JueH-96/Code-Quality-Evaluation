# YOUR CODE HERE
def dfs(grid, visited, i, j, h, w):
    if i < 0 or i >= h or j < 0 or j >= w:
        return
    if visited[i][j] or grid[i][j] == '.':
        return
    
    visited[i][j] = True
    
    # Check all 8 adjacent cells (horizontal, vertical, and diagonal)
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for di, dj in directions:
        ni, nj = i + di, j + dj
        dfs(grid, visited, ni, nj, h, w)

h, w = map(int, input().split())
grid = []
for _ in range(h):
    grid.append(input())

visited = [[False] * w for _ in range(h)]
count = 0

for i in range(h):
    for j in range(w):
        if grid[i][j] == '#' and not visited[i][j]:
            dfs(grid, visited, i, j, h, w)
            count += 1

print(count)