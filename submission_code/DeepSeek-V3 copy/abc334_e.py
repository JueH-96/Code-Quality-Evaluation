import sys
from collections import deque

MOD = 998244353

def main():
    H, W = map(int, sys.stdin.readline().split())
    grid = [list(sys.stdin.readline().strip()) for _ in range(H)]
    
    # Count the number of red cells
    red_cells = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                red_cells.append((i, j))
    R = len(red_cells)
    
    # Precompute the number of green connected components in the original grid
    original_components = 0
    visited = [[False for _ in range(W)] for _ in range(H)]
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#' and not visited[i][j]:
                original_components += 1
                q = deque()
                q.append((i, j))
                visited[i][j] = True
                while q:
                    x, y = q.popleft()
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] == '#' and not visited[nx][ny]:
                            visited[nx][ny] = True
                            q.append((nx, ny))
    
    # For each red cell, compute the change in the number of components when it is painted green
    total = 0
    for i, j in red_cells:
        # Temporarily paint the cell green
        grid[i][j] = '#'
        # Compute the new number of components
        new_components = 0
        visited_new = [[False for _ in range(W)] for _ in range(H)]
        for x in range(H):
            for y in range(W):
                if grid[x][y] == '#' and not visited_new[x][y]:
                    new_components += 1
                    q = deque()
                    q.append((x, y))
                    visited_new[x][y] = True
                    while q:
                        cx, cy = q.popleft()
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] == '#' and not visited_new[nx][ny]:
                                visited_new[nx][ny] = True
                                q.append((nx, ny))
        # Revert the cell back to red
        grid[i][j] = '.'
        # Add the new number of components to the total
        total += new_components
    
    # Compute the expected value as (total) / R
    # Since we need to compute (total / R) mod MOD, we need to find the modular inverse of R
    # Compute R_inv = pow(R, MOD-2, MOD)
    R_inv = pow(R, MOD-2, MOD)
    expected = (total * R_inv) % MOD
    print(expected)

if __name__ == "__main__":
    main()