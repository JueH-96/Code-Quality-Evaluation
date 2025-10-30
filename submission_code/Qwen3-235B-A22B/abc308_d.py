import sys
from collections import deque

def main():
    H, W = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    if grid[0][0] != 's':
        print("No")
        return

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    snuke = 'snuke'
    visited = [[[False] * 5 for _ in range(W)] for _ in range(H)]
    visited[0][0][0] = True

    q = deque()
    q.append((0, 0, 0))

    while q:
        i, j, s = q.popleft()
        if i == H - 1 and j == W - 1:
            print("Yes")
            return

        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                new_s = (s + 1) % 5
                if grid[ni][nj] == snuke[new_s] and not visited[ni][nj][new_s]:
                    visited[ni][nj][new_s] = True
                    q.append((ni, nj, new_s))

    print("No")

if __name__ == "__main__":
    main()