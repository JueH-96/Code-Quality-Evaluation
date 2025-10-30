import sys
from collections import deque
from array import array

def main() -> None:
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    grid = [sys.stdin.readline().rstrip() for _ in range(N)]

    # Directions: up, down, left, right
    DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Flatten (r, c) to single id  0 … V-1
    def idx(r: int, c: int) -> int:
        return r * N + c

    V = N * N                           # number of cells (<= 3600)

    # Locate the two players
    players = []
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'P':
                players.append(idx(i, j))
    s1, s2 = players                   # starting positions
    if s1 == s2:                       # already in the same cell (should not happen by statement)
        print(0)
        return

    # Pre–compute, for every cell and for every direction, where the player ends up
    nxt = [[0] * V for _ in range(4)]  # nxt[d][v] -> destination id when moving to d
    for r in range(N):
        for c in range(N):
            cur = idx(r, c)
            for d, (dr, dc) in enumerate(DIRS):
                nr, nc = r + dr, c + dc
                if 0 <= nr < N and 0 <= nc < N and grid[nr][nc] != '#':
                    nxt[d][cur] = idx(nr, nc)
                else:
                    nxt[d][cur] = cur          # blocked → stay put

    # Total number of possible ordered pairs of positions
    TOT = V * V                              # <= 12 960 000  (N ≤ 60)

    # Bit-set for visited pairs
    vis = bytearray((TOT + 7) // 8)

    def visited(k: int) -> bool:
        return (vis[k >> 3] >> (k & 7)) & 1

    def mark(k: int) -> None:
        vis[k >> 3] |= 1 << (k & 7)

    start_key = s1 * V + s2
    mark(start_key)

    # Frontier BFS (layer by layer) storing only current and next layer.
    cur_layer = array('I', [start_key])
    steps = 0

    while cur_layer:
        steps += 1
        nxt_layer = array('I')
        for key in cur_layer:
            p1 = key // V
            p2 = key - p1 * V          # faster than %

            for d in range(4):
                np1 = nxt[d][p1]
                np2 = nxt[d][p2]

                if np1 == np2:         # players meet
                    print(steps)
                    return

                nk = np1 * V + np2
                if not visited(nk):
                    mark(nk)
                    nxt_layer.append(nk)
        cur_layer = nxt_layer

    print(-1)


if __name__ == "__main__":
    main()