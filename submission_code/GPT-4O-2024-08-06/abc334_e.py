# YOUR CODE HERE
import sys
from collections import deque

MOD = 998244353

def mod_inv(x, mod):
    # Fermat's little theorem: x^(mod-2) is the modular inverse of x % mod
    return pow(x, mod - 2, mod)

def count_components(grid, H, W):
    visited = [[False] * W for _ in range(H)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def bfs(start_i, start_j):
        queue = deque([(start_i, start_j)])
        visited[start_i][start_j] = True
        while queue:
            i, j = queue.popleft()
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W and not visited[ni][nj] and grid[ni][nj] == '#':
                    visited[ni][nj] = True
                    queue.append((ni, nj))
    
    component_count = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#' and not visited[i][j]:
                component_count += 1
                bfs(i, j)
    
    return component_count

def main():
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    grid = data[2:H+2]
    
    initial_components = count_components(grid, H, W)
    
    red_cells = [(i, j) for i in range(H) for j in range(W) if grid[i][j] == '.']
    num_red_cells = len(red_cells)
    
    sum_components = 0
    
    for i, j in red_cells:
        new_grid = [list(row) for row in grid]
        new_grid[i][j] = '#'
        new_components = count_components(new_grid, H, W)
        sum_components += new_components
    
    expected_value_numerator = sum_components
    expected_value_denominator = num_red_cells
    
    result = (expected_value_numerator * mod_inv(expected_value_denominator, MOD)) % MOD
    
    print(result)

if __name__ == "__main__":
    main()