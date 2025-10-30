import sys
from collections import deque

def main() -> None:
    sys.setrecursionlimit(1 << 25)
    H, W = map(int, sys.stdin.readline().split())
    S = [sys.stdin.readline().rstrip() for _ in range(H)]
    A, B, C, D = map(int, sys.stdin.readline().split())
    A -= 1
    B -= 1
    C -= 1
    D -= 1

    INF = 10 ** 9
    dist = [[INF] * W for _ in range(H)]
    dist[A][B] = 0
    dq = deque()
    dq.append((A, B))

    # directions : up, down, left, right
    DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while dq:
        x, y = dq.popleft()
        d = dist[x][y]
        if d > dist[x][y]:   # (rarely needed because we never push worse distance to left)
            continue

        # 0-cost moves to adjacent road cells
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W:
                # We can move with cost 0 if the destination is already a road
                # (either from the start or made a road previously)
                if (S[nx][ny] == '.' or dist[nx][ny] != INF) and dist[nx][ny] > d:
                    dist[nx][ny] = d
                    dq.appendleft((nx, ny))

        # 1-cost moves produced by a front kick
        for dx, dy in DIRS:
            for k in (1, 2):
                nx, ny = x + dx * k, y + dy * k
                if not (0 <= nx < H and 0 <= ny < W):
                    break          # outside the grid – nothing further in this direction
                if dist[nx][ny] > d + 1:
                    dist[nx][ny] = d + 1
                    dq.append((nx, ny))

    print(dist[C][D])

if __name__ == "__main__":
    main()