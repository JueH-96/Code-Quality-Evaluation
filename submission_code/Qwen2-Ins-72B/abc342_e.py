from heapq import heappush, heappop
from collections import defaultdict

def solve(N, M, train_info):
    # Build graph
    graph = defaultdict(list)
    for l, d, k, c, A, B in train_info:
        for i in range(k):
            t = l + i * d
            graph[B].append((t + c, A))

    # Bellman-Ford algorithm
    dist = [float('inf')] * N
    dist[N-1] = 0
    for _ in range(N):
        for B in range(N):
            for t, A in graph[B]:
                if dist[A] > dist[B] and t <= dist[B]:
                    dist[A] = dist[B]
                    if _ == N - 1:
                        dist[A] = t

    # Output results
    for i in range(N-1):
        if dist[i] == float('inf'):
            print("Unreachable")
        else:
            print(dist[i])

N, M = map(int, input().split())
train_info = [list(map(int, input().split())) for _ in range(M)]
solve(N, M, train_info)