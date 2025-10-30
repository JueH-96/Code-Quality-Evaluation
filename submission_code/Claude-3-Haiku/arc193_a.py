from collections import defaultdict
from heapq import heappush, heappop

def find_min_weight_path(N, W, L_R, s, t):
    graph = defaultdict(list)
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            if not (L_R[i-1][0] <= L_R[j-1][1] and L_R[j-1][0] <= L_R[i-1][1]):
                graph[i].append(j)
                graph[j].append(i)

    if s not in graph or t not in graph:
        return -1

    pq = [(0, s)]
    dist = [float('inf')] * (N+1)
    dist[s] = 0

    while pq:
        d, u = heappop(pq)
        if u == t:
            return d
        if d > dist[u]:
            continue
        for v in graph[u]:
            new_dist = d + W[v-1]
            if new_dist < dist[v]:
                dist[v] = new_dist
                heappush(pq, (new_dist, v))

    return -1

# Read input
N = int(input())
W = [int(x) for x in input().split()]
L_R = []
for _ in range(N):
    l, r = [int(x) for x in input().split()]
    L_R.append((l, r))

Q = int(input())
for _ in range(Q):
    s, t = [int(x) for x in input().split()]
    print(find_min_weight_path(N, W, L_R, s, t))