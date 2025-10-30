from collections import deque
import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(10**7)
    H, W = map(int, sys.stdin.readline().split())
    S = [sys.stdin.readline().rstrip() for _ in range(H)]
    A, B, C, D = map(int, sys.stdin.readline().split())
    # convert to 0-based
    A -= 1; B -= 1; C -= 1; D -= 1

    INF = 10**18
    dist = [[INF]*W for _ in range(H)]
    dq = deque()
    dist[A][B] = 0
    dq.append((A, B))

    # Directions for zero-cost moves (adjacent roads)
    adj_dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    # Directions for kick (straight lines)
    kick_dirs = [(-1,0),(1,0),(0,-1),(0,1)]

    while dq:
        i, j = dq.popleft()
        d0 = dist[i][j]
        # 0-cost moves into adjacent road cells
        for dx, dy in adj_dirs:
            ni, nj = i+dx, j+dy
            if 0 <= ni < H and 0 <= nj < W:
                if S[ni][nj]=='.' and dist[ni][nj] > d0:
                    dist[ni][nj] = d0
                    dq.appendleft((ni, nj))
        # 1-cost moves: "kick and move" up to 2 cells in straight dir
        for dx, dy in kick_dirs:
            for step in (1,2):
                ni, nj = i + dx*step, j + dy*step
                if 0 <= ni < H and 0 <= nj < W:
                    if dist[ni][nj] > d0 + 1:
                        dist[ni][nj] = d0 + 1
                        dq.append((ni, nj))

    ans = dist[C][D]
    print(ans)

if __name__ == "__main__":
    main()