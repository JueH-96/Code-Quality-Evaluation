import sys
from collections import deque

MOD = 998244353

def main():
    H, W = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    component = [[-1 for _ in range(W)] for _ in range(H)]
    current = 0

    # Assign component IDs to green cells
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#' and component[i][j] == -1:
                queue = deque()
                queue.append((i, j))
                component[i][j] = current
                while queue:
                    x, y = queue.popleft()
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < H and 0 <= ny < W:
                            if grid[nx][ny] == '#' and component[nx][ny] == -1:
                                component[nx][ny] = current
                                queue.append((nx, ny))
                current += 1

    C = current
    red_cells = 0
    sum_contribution = 0

    # Calculate contributions from each red cell
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                red_cells += 1
                unique_components = set()
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = i + dx, j + dy
                    if 0 <= nx < H and 0 <= ny < W:
                        if grid[nx][ny] == '#':
                            unique_components.add(component[nx][ny])
                K = len(unique_components)
                sum_contribution += (C + 1 - K)

    # Compute the result modulo 998244353
    inv_red = pow(red_cells, MOD - 2, MOD)
    ans = (sum_contribution % MOD) * inv_red % MOD
    print(ans)

if __name__ == "__main__":
    main()