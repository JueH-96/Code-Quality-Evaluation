import sys
from collections import deque

def main() -> None:
    sys.setrecursionlimit(1 << 25)
    H, W, T = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]

    start = goal = None
    candies = []

    for r in range(H):
        for c, ch in enumerate(grid[r]):
            if ch == 'S':
                start = (r, c)
            elif ch == 'G':
                goal = (r, c)
            elif ch == 'o':
                candies.append((r, c))

    # list of every special node : 0 -> S, 1..C -> candies, C+1 -> G
    nodes = [start] + candies + [goal]
    C = len(candies)
    K = C + 2

    # distance matrix between special nodes
    dist = [[-1] * K for _ in range(K)]
    moves = ((1, 0), (-1, 0), (0, 1), (0, -1))

    def bfs(idx: int) -> None:
        sr, sc = nodes[idx]
        d = [[-1] * W for _ in range(H)]
        dq = deque([(sr, sc)])
        d[sr][sc] = 0
        while dq:
            r, c = dq.popleft()
            for dr, dc in moves:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if grid[nr][nc] == '#' or d[nr][nc] != -1:
                    continue
                d[nr][nc] = d[r][c] + 1
                dq.append((nr, nc))
        for j in range(K):
            rj, cj = nodes[j]
            dist[idx][j] = d[rj][cj]

    for i in range(K):
        bfs(i)

    shortest_SG = dist[0][K - 1]
    if shortest_SG == -1 or shortest_SG > T:
        print(-1)
        return
    if C == 0:                # no candies on the board
        print(0)
        return

    INF = 10 ** 9
    dp = [[INF] * C for _ in range(1 << C)]   # dp[mask][last] = min length
    for i in range(C):
        d = dist[0][1 + i]
        if d != -1:
            dp[1 << i][i] = d

    for mask in range(1 << C):
        for last in range(C):
            cur = dp[mask][last]
            if cur == INF:
                continue
            for nxt in range(C):
                if mask & (1 << nxt):
                    continue
                d = dist[1 + last][1 + nxt]
                if d == -1:
                    continue
                nmask = mask | (1 << nxt)
                nd = cur + d
                if nd < dp[nmask][nxt]:
                    dp[nmask][nxt] = nd

    best = 0   # at least we can always reach G directly (we already checked)
    for mask in range(1 << C):
        cnt = bin(mask).count('1')
        for last in range(C):
            cur = dp[mask][last]
            if cur == INF:
                continue
            g = dist[1 + last][K - 1]
            if g == -1:
                continue
            if cur + g <= T and cnt > best:
                best = cnt

    print(best)

if __name__ == "__main__":
    main()