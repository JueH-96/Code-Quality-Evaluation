import sys
from collections import deque

def main():
    MOD = 998244353
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    H = int(data[ptr])
    ptr += 1
    W = int(data[ptr])
    ptr += 1
    grid = []
    for _ in range(H):
        grid.append(data[ptr])
        ptr += 1
    
    comp_id = [[0] * W for _ in range(H)]
    C_initial = 0
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#' and comp_id[i][j] == 0:
                queue = deque()
                queue.append((i, j))
                comp_id[i][j] = C_initial + 1
                while queue:
                    x, y = queue.popleft()
                    for dx, dy in directions:
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < H and 0 <= ny < W:
                            if grid[nx][ny] == '#' and comp_id[nx][ny] == 0:
                                comp_id[nx][ny] = C_initial + 1
                                queue.append((nx, ny))
                C_initial += 1
    
    R = sum(row.count('.') for row in grid)
    total = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                adjacent_comps = set()
                for dx, dy in directions:
                    x = i + dx
                    y = j + dy
                    if 0 <= x < H and 0 <= y < W:
                        if grid[x][y] == '#':
                            adjacent_comps.add(comp_id[x][y])
                k = len(adjacent_comps)
                total += (C_initial + 1 - k)
    
    inv_R = pow(R, MOD - 2, MOD)
    ans = (total * inv_R) % MOD
    print(ans)

if __name__ == "__main__":
    main()