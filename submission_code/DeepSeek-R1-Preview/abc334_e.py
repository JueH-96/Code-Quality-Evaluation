import sys
from collections import deque

MOD = 998244353

def main():
    H, W = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    
    component = [[0] * W for _ in range(H)]
    current_component = 0
    
    # Assign component IDs to green cells using BFS
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#' and component[i][j] == 0:
                current_component += 1
                q = deque()
                q.append((i, j))
                component[i][j] = current_component
                while q:
                    x, y = q.popleft()
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < H and 0 <= ny < W:
                            if grid[nx][ny] == '#' and component[nx][ny] == 0:
                                component[nx][ny] = current_component
                                q.append((nx, ny))
    
    C = current_component
    sum_contrib = 0
    red_count = 0
    
    # Calculate contributions for each red cell
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                red_count += 1
                adj_components = set()
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx = i + dx
                    ny = j + dy
                    if 0 <= nx < H and 0 <= ny < W:
                        if grid[nx][ny] == '#':
                            adj_components.add(component[nx][ny])
                k_i = len(adj_components)
                sum_contrib += (C + 1 - k_i)
    
    # Compute the result modulo 998244353
    if red_count == 0:
        print(0)
    else:
        numerator = sum_contrib % MOD
        denominator = red_count
        inv_denominator = pow(denominator, MOD - 2, MOD)
        result = (numerator * inv_denominator) % MOD
        print(result)

if __name__ == "__main__":
    main()