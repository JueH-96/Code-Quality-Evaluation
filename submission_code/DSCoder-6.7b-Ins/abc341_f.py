import sys
from heapq import *

def solve():
    N, M = map(int, sys.stdin.readline().split())
    graph = [[] for _ in range(N+1)]
    weights = [0] * (N+1)
    pieces = [0] * (N+1)
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        graph[u].append(v)
        graph[v].append(u)
    weights = list(map(int, sys.stdin.readline().split()))
    pieces = list(map(int, sys.stdin.readline().split()))

    max_heap = []
    for i in range(1, N+1):
        if pieces[i] > 0:
            heappush(max_heap, (-weights[i], i))

    total_pieces = 0
    while max_heap:
        w, u = heappop(max_heap)
        w = -w
        if pieces[u] == 0:
            continue
        total_pieces += 1
        pieces[u] -= 1
        for v in graph[u]:
            if pieces[v] > 0:
                pieces[v] -= 1
                heappush(max_heap, (-weights[v], v))

    print(total_pieces)

solve()