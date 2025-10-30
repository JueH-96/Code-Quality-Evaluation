import sys
from collections import deque

MOD = 998244353

def main():
    H, W = map(int, sys.stdin.readline().split())
    grid = [list(sys.stdin.readline().strip()) for _ in range(H)]
    comp = [[-1 for _ in range(W)] for _ in range(H)]
    current_id = 0

    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#' and comp[i][j] == -1:
                current_id += 1
                q = deque()
                q.append((i, j))
                comp[i][j] = current_id
                while q:
                    x, y = q.popleft()
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < H and 0 <= ny < W:
                            if grid[nx][ny] == '#' and comp[nx][ny] == -1:
                                comp[nx][ny] = current_id
                                q.append((nx, ny))

    total = 0
    R = 0

    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                R += 1
                adjacent_comps = set()
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni = i + dx
                    nj = j + dy
                    if 0 <= ni < H and 0 <= nj < W:
                        if grid[ni][nj] == '#':
                            adjacent_comps.add(comp[ni][nj])
                m = len(adjacent_comps)
                contribution = (current_id - m + 1)
                total += contribution

    if R == 0:
        print(0)
        return

    inv_R = pow(R, MOD - 2, MOD)
    result = (total * inv_R) % MOD
    print(result)

if __name__ == '__main__':
    main()