import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    graph = [[] for _ in range(N+1)]
    weights = [0] * (N+1)
    for _ in range(M):
        u, v, w = map(int, sys.stdin.readline().split())
        graph[u].append((v, w))
        graph[v].append((u, w))
        weights[u] += w
        weights[v] += w
    for u in range(1, N+1):
        for v, w in graph[u]:
            if weights[u] + w < weights[v]:
                weights[v] = weights[u] + w
    min_weight = min(weights[1:])
    if min_weight < 0:
        print("No")
    else:
        print(min_weight)

solve()