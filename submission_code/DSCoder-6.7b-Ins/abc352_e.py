import sys
from heapq import heappop, heappush

def read_input():
    N, M = map(int, sys.stdin.readline().split())
    operations = []
    for _ in range(M):
        K, C = map(int, sys.stdin.readline().split())
        vertices = list(map(int, sys.stdin.readline().split()))
        operations.append((K, C, vertices))
    return N, M, operations

def solve():
    N, M, operations = read_input()
    graph = [[] for _ in range(N+1)]
    for K, C, vertices in operations:
        for i in range(K):
            for j in range(i+1, K):
                u, v = vertices[i], vertices[j]
                graph[u].append((v, C))
                graph[v].append((u, C))
    visited = [False] * (N+1)
    queue = [(0, 1)]  # start from vertex 1
    total_weight = 0
    while queue:
        weight, u = heappop(queue)
        if not visited[u]:
            visited[u] = True
            total_weight += weight
            for v, weight in graph[u]:
                if not visited[v]:
                    heappush(queue, (weight, v))
    if all(visited):
        print(total_weight)
    else:
        print(-1)

solve()