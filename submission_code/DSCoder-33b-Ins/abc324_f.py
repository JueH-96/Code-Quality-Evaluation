# YOUR CODE HERE
import sys
from heapq import heappop, heappush

def solve():
    n, m = map(int, sys.stdin.readline().split())
    graph = [[] for _ in range(n+1)]
    for _ in range(m):
        u, v, b, c = map(int, sys.stdin.readline().split())
        graph[u].append((v, b, c))

    dist = [0] + [-float('inf')]*n
    heap = [(0, 0, 1)]
    while heap:
        beauty, cost, node = heappop(heap)
        beauty = -beauty
        cost = -cost
        if dist[node] >= beauty/cost:
            continue
        dist[node] = beauty/cost
        for v, b, c in graph[node]:
            heappush(heap, (beauty-b, cost-c, v))
    print(dist[n])

solve()