import sys
from heapq import heappop, heappush

def solve():
    N, Q = map(int, sys.stdin.readline().split())
    edges = [list(map(int, sys.stdin.readline().split())) for _ in range(Q)]
    edges.sort(key=lambda x: x[2])
    parent = list(range(N + 1))
    cost = 0
    for L, R, C in edges:
        i, j = L, R
        while i != parent[i]:
            i = parent[i]
        while j != parent[j]:
            j = parent[j]
        if i != j:
            cost += C
            parent[j] = i
    for i in range(1, N + 1):
        if parent[i] != i:
            return -1
    return cost

print(solve())