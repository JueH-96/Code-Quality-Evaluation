# YOUR CODE HERE
import sys
from collections import deque

MOD = 998244353

def modinv(x, mod=MOD):
    return pow(x, mod-2, mod)

def bfs(grid, visited, start, H, W):
    queue = deque([start])
    visited[start[0]][start[1]] = True
    while queue:
        x, y = queue.popleft()
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and not visited[nx][ny] and grid[nx][ny] == '#':
                visited[nx][ny] = True
                queue.append((nx, ny))

def count_components(grid, H, W):
    visited = [[False] * W for _ in range(H)]
    components = 0
    for i in range(H):
        for j in range(W):
            if not visited[i][j] and grid[i][j] == '#':
                bfs(grid, visited, (i, j), H, W)
                components += 1
    return components

def solve(H, W, grid):
    red_cells = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                red_cells.append((i, j))
    
    total_red = len(red_cells)
    total_components = 0
    
    for i, j in red_cells:
        grid[i][j] = '#'
        total_components += count_components(grid, H, W)
        grid[i][j] = '.'
    
    expected_value = (total_components * modinv(total_red, MOD)) % MOD
    return expected_value

# Read input
H, W = map(int, input().split())
grid = [list(input().strip()) for _ in range(H)]

# Solve and print the result
print(solve(H, W, grid))