import sys
import threading
from collections import deque

def main():
    input = sys.stdin.readline
    H, W = map(int, input().split())
    grid = [input().rstrip() for _ in range(H)]
    A, B, C, D = map(int, input().split())
    # convert to 0-based
    A -= 1; B -= 1; C -= 1; D -= 1

    INF = 10**9
    dist = [[INF]*W for _ in range(H)]
    dist[A][B] = 0

    dq = deque()
    dq.append((A, B))

    # directions for zero-cost moves
    moves = [(-1,0), (1,0), (0,-1), (0,1)]

    while dq:
        i, j = dq.popleft()
        d = dist[i][j]
        # if we've reached the target, we can stop
        if i == C and j == D:
            break
        # 0-cost moves to adjacent roads
        for di, dj in moves:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                if grid[ni][nj] == '.' and dist[ni][nj] > d:
                    dist[ni][nj] = d
                    dq.appendleft((ni, nj))
        # cost-1 "front kick" moves: reach cells 1 or 2 steps away in straight line
        nd = d + 1
        for di, dj in moves:
            # step = 1
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and dist[ni][nj] > nd:
                dist[ni][nj] = nd
                dq.append((ni, nj))
            # step = 2
            ni2, nj2 = i + 2*di, j + 2*dj
            if 0 <= ni2 < H and 0 <= nj2 < W and dist[ni2][nj2] > nd:
                dist[ni2][nj2] = nd
                dq.append((ni2, nj2))

    # print the minimum kicks to reach (C,D)
    print(dist[C][D])

if __name__ == "__main__":
    main()