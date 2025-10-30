# YOUR CODE HERE
import sys

MOD = 998244353

def mod_inverse(x):
    return pow(x, MOD - 2, MOD)

def find_components(grid):
    H, W = len(grid), len(grid[0])
    visited = [[False] * W for _ in range(H)]
    components = 0

    def dfs(i, j):
        if i < 0 or i >= H or j < 0 or j >= W or visited[i][j] or grid[i][j] == '.':
            return
        visited[i][j] = True
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            dfs(i + di, j + dj)

    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#' and not visited[i][j]:
                dfs(i, j)
                components += 1

    return components

def solve():
    H, W = map(int, input().split())
    grid = [input().strip() for _ in range(H)]

    initial_components = find_components(grid)
    red_cells = sum(row.count('.') for row in grid)

    total_new_components = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                new_grid = [list(row) for row in grid]
                new_grid[i][j] = '#'
                new_components = find_components(new_grid)
                total_new_components += new_components

    expected_value = (total_new_components - initial_components * red_cells) * mod_inverse(red_cells) % MOD
    print(expected_value)

solve()