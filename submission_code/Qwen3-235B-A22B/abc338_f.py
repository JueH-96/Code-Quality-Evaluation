import sys
from collections import deque

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1

    edges = [[] for _ in range(N + 1)]
    for _ in range(M):
        u = int(input[idx])
        idx += 1
        v = int(input[idx])
        idx += 1
        w = int(input[idx])
        idx += 1
        edges[u].append((v, w))

    INF = (1 << 60)
    mask_all = (1 << N) - 1

    # Initialize distance array
    dist = [ [INF] * (N + 1) for _ in range(1 << N) ]
    in_queue = [ [False] * (N + 1) for _ in range(1 << N) ]

    q = deque()

    for u in range(1, N + 1):
        mask = 1 << (u - 1)
        dist[mask][u] = 0
        q.append((mask, u))
        in_queue[mask][u] = True

    while q:
        mask, u = q.popleft()
        in_queue[mask][u] = False
        for (v, w) in edges[u]:
            new_mask = mask | (1 << (v - 1))
            new_dist = dist[mask][u] + w
            if new_dist < dist[new_mask][v]:
                dist[new_mask][v] = new_dist
                if not in_queue[new_mask][v]:
                    q.append((new_mask, v))
                    in_queue[new_mask][v] = True

    ans = INF
    for u in range(1, N + 1):
        ans = min(ans, dist[mask_all][u])

    if ans == INF:
        print("No")
    else:
        print(ans)

if __name__ == "__main__":
    main()