import sys
from collections import deque

MOD = 998244353

def main():
    H, W = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    
    comp_id = [[-1 for _ in range(W)] for _ in range(H)]
    current_id = 0
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#' and comp_id[i][j] == -1:
                queue = deque()
                queue.append((i, j))
                comp_id[i][j] = current_id
                while queue:
                    x, y = queue.popleft()
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < H and 0 <= ny < W:
                            if grid[nx][ny] == '#' and comp_id[nx][ny] == -1:
                                comp_id[nx][ny] = current_id
                                queue.append((nx, ny))
                current_id += 1
    
    C = current_id
    R = sum(row.count('.') for row in grid)
    
    sum_total = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                components = set()
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni = i + dx
                    nj = j + dy
                    if 0 <= ni < H and 0 <= nj < W:
                        if grid[ni][nj] == '#':
                            components.add(comp_id[ni][nj])
                k = len(components)
                term = (C + 1 - k) % MOD
                sum_total = (sum_total + term) % MOD
    
    inv_R = pow(R, MOD - 2, MOD) if R != 0 else 0
    ans = (sum_total * inv_R) % MOD
    print(ans)

if __name__ == "__main__":
    main()