# YOUR CODE HERE
import sys
from collections import deque

MOD = 998244353

def find_connected_components(grid, H, W):
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

def mod_inverse(a, m):
    return pow(a, m - 2, m)

def main():
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    grid = [list(data[i + 2]) for i in range(H)]
    
    initial_components = find_connected_components(grid, H, W)
    
    red_cells = [(i, j) for i in range(H) for j in range(W) if grid[i][j] == '.']
    total_red_cells = len(red_cells)
    
    total_components = 0
    for i, j in red_cells:
        grid[i][j] = '#'
        new_components = find_connected_components(grid, H, W)
        total_components += new_components
        grid[i][j] = '.'
    
    P = total_components
    Q = total_red_cells
    
    Q_inv = mod_inverse(Q, MOD)
    result = (P * Q_inv) % MOD
    
    print(result)

if __name__ == "__main__":
    main()