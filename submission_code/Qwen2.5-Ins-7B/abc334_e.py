# YOUR CODE HERE
from collections import defaultdict
from math import comb

def solve():
    mod = 998244353
    H, W = map(int, input().split())
    grid = [input() for _ in range(H)]
    green_cells = [(i, j) for i in range(H) for j in range(W) if grid[i][j] == '#']
    red_cells = [(i, j) for i in range(H) for j in range(W) if grid[i][j] == '.']
    
    def get_neighbors(cell):
        i, j = cell
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == '#':
                yield (ni, nj)
    
    def count_components():
        visited = set()
        components = 0
        for cell in green_cells:
            if cell not in visited:
                components += 1
                stack = [cell]
                while stack:
                    i, j = stack.pop()
                    visited.add((i, j))
                    for neighbor in get_neighbors((i, j)):
                        if neighbor not in visited:
                            stack.append(neighbor)
        return components
    
    total_components = count_components()
    dp = defaultdict(int)
    for cell in green_cells:
        dp[cell] = 1
        for neighbor in get_neighbors(cell):
            dp[cell] += dp[neighbor]
    
    expected_value = 0
    for cell in red_cells:
        new_components = total_components - 1 + dp[cell]
        expected_value += new_components
        expected_value %= mod
    
    print(expected_value * pow(len(red_cells), mod-2, mod) % mod)

solve()