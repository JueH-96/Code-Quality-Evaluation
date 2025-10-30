import sys
from collections import deque

def main():
    input = sys.stdin.readline
    H, W = map(int, input().split())
    grid = [input().rstrip() for _ in range(H)]
    A, B, C, D = map(int, input().split())
    # convert to 0-based
    A -= 1; B -= 1; C -= 1; D -= 1

    INF = 10**9
    N = H * W
    dist = [INF] * N

    start = A * W + B
    target = C * W + D
    dist[start] = 0

    dq = deque([start])
    # four cardinal directions
    dr = (-1, 1, 0, 0)
    dc = ( 0, 0,-1, 1)

    while dq:
        pos = dq.popleft()
        # once we pop the target, its dist is minimal
        if pos == target:
            break
        d = dist[pos]
        i = pos // W
        j = pos % W

        # 0-cost moves to adjacent roads
        for k in range(4):
            ni = i + dr[k]
            nj = j + dc[k]
            if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == '.':
                npos = ni * W + nj
                if dist[npos] > d:
                    dist[npos] = d
                    dq.appendleft(npos)

        # cost-1 "kick" moves up to 2 steps in each cardinal direction
        nd = d + 1
        for k in range(4):
            drk = dr[k]
            dck = dc[k]
            # one step
            ni = i + drk
            nj = j + dck
            if 0 <= ni < H and 0 <= nj < W:
                npos = ni * W + nj
                if dist[npos] > nd:
                    dist[npos] = nd
                    dq.append(npos)
            # two steps
            ni2 = i + drk*2
            nj2 = j + dck*2
            if 0 <= ni2 < H and 0 <= nj2 < W:
                npos2 = ni2 * W + nj2
                if dist[npos2] > nd:
                    dist[npos2] = nd
                    dq.append(npos2)

    # print minimum number of kicks to reach target
    print(dist[target])

if __name__ == "__main__":
    main()