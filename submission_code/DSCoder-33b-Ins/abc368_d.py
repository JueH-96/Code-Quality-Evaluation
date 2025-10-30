import sys
from collections import defaultdict
from heapq import heappop, heappush

def solve():
    n, k = map(int, input().split())
    edges = defaultdict(list)
    for _ in range(n-1):
        a, b = map(int, input().split())
        edges[a].append(b)
        edges[b].append(a)
    vertices = list(map(int, input().split()))
    vertices.sort()
    visited = [False] * (n+1)
    heap = []
    for v in vertices:
        heappush(heap, (0, v))
        visited[v] = True
    while heap:
        d, v = heappop(heap)
        for u in edges[v]:
            if not visited[u]:
                heappush(heap, (d+1, u))
                visited[u] = True
    print(sum(visited))

solve()