# YOUR CODE HERE

import sys
import threading

def main():
    import sys
    import heapq

    sys.setrecursionlimit(1 << 25)
    N, A, B, C = map(int, sys.stdin.readline().split())
    N = int(N)
    D = [list(map(int, sys.stdin.readline().split())) for _ in range(N)]

    adj = [[] for _ in range(2 * N)]  # Nodes 0 to N-1: car mode, N to 2N-1: train mode

    for i in range(N):
        # Edge to switch from car mode to train mode at city i
        adj[i].append((i + N, 0))

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = D[i][j]
            # Edge from i to j in car mode
            adj[i].append((j, d * A))
            # Edge from i+N to j+N in train mode
            adj[i + N].append((j + N, d * B + C))

    INF = 1 << 60
    dist = [INF] * (2 * N)
    dist[0] = 0  # Start at city 0 in car mode
    hq = [(0, 0)]  # (cost, node)

    while hq:
        cost, node = heapq.heappop(hq)
        if dist[node] < cost:
            continue
        for neighbor, edge_cost in adj[node]:
            new_cost = cost + edge_cost
            if dist[neighbor] > new_cost:
                dist[neighbor] = new_cost
                heapq.heappush(hq, (new_cost, neighbor))

    ans = min(dist[N - 1], dist[2 * N -1])

    print(ans)

threading.Thread(target=main).start()