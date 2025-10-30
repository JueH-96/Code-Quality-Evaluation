import sys
from collections import defaultdict, deque

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

bridges = defaultdict(list)
for i in range(M):
    u = int(data[2 + 3 * i]) - 1
    v = int(data[3 + 3 * i]) - 1
    t = int(data[4 + 3 * i])
    bridges[u].append((v, t))
    bridges[v].append((u, t))

Q = int(data[2 + 3 * M])

results = []

for q in range(Q):
    K = int(data[3 * M + 2 * q + 2])
    B = list(map(int, data[3 * M + 2 * q + 3: 3 * M + 2 * q + 3 + K]))
    B = [b - 1 for b in B]

    # Convert bridge indices to actual bridge data
    B_data = [bridges[b] for b in B]

    # Use Dijkstra's algorithm to find the shortest path from island 1 to island N
    dist = [float('inf')] * N
    dist[0] = 0
    pq = [(0, 0)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, t in bridges[u]:
            if dist[u] + t < dist[v]:
                dist[v] = dist[u] + t
                heapq.heappush(pq, (dist[v], v))

    # Use dynamic programming to find the minimum time required to travel from island 1 to island N using each of the given bridges at least once
    dp = [float('inf')] * (1 << K)
    dp[0] = 0
    for mask in range(1 << K):
        for i in range(K):
            if mask & (1 << i):
                dp[mask] = min(dp[mask], dp[mask ^ (1 << i)] + B_data[i][0][1])

    results.append(dist[N - 1] + dp[(1 << K) - 1])

for result in results:
    print(result)