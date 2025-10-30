import sys
from collections import deque

def main():
    H, W = map(int, sys.stdin.readline().split())
    grid = []
    for _ in range(H):
        s = sys.stdin.readline().strip()
        grid.append(list(s))
    
    component_id = [[-1 for _ in range(W)] for _ in range(H)]
    current_component = 0

    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#' and component_id[i][j] == -1:
                q = deque()
                q.append((i, j))
                component_id[i][j] = current_component
                while q:
                    x, y = q.popleft()
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < H and 0 <= ny < W:
                            if grid[nx][ny] == '#' and component_id[nx][ny] == -1:
                                component_id[nx][ny] = current_component
                                q.append((nx, ny))
                current_component += 1

    original_components = current_component
    total_red = 0

    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                total_red += 1

    sum_deltas = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                s = set()
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx = i + dx
                    ny = j + dy
                    if 0 <= nx < H and 0 <= ny < W:
                        if grid[nx][ny] == '#':
                            s.add(component_id[nx][ny])
                m = len(s)
                delta = 1 - m
                sum_deltas += delta

    MOD = 998244353
    numerator = original_components * total_red + sum_deltas
    denominator = total_red
    inv_denominator = pow(denominator, MOD - 2, MOD)
    ans = (numerator % MOD) * inv_denominator % MOD
    print(ans)

if __name__ == '__main__':
    main()