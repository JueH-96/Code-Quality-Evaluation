import sys
from collections import deque

MOD = 998244353

def main():
    H, W = map(int, sys.stdin.readline().split())
    grid = [list(sys.stdin.readline().strip()) for _ in range(H)]
    
    component = [[-1 for _ in range(W)] for _ in range(H)]
    current = 0
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#' and component[i][j] == -1:
                queue = deque()
                queue.append((i, j))
                component[i][j] = current
                while queue:
                    x, y = queue.popleft()
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < H and 0 <= ny < W:
                            if grid[nx][ny] == '#' and component[nx][ny] == -1:
                                component[nx][ny] = current
                                queue.append((nx, ny))
                current += 1
    
    C_initial = current
    sum_K = 0
    R = 0
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                R += 1
                comps = set()
                for dx, dy in directions:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < H and 0 <= nj < W:
                        if grid[ni][nj] == '#':
                            comps.add(component[ni][nj])
                sum_K += len(comps)
    
    numerator = (C_initial * R + (R - sum_K)) % MOD
    denominator = R
    inv_denominator = pow(denominator, MOD - 2, MOD)
    result = (numerator * inv_denominator) % MOD
    print(result)

if __name__ == '__main__':
    main()