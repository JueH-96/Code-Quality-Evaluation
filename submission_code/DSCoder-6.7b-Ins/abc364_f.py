import sys
from heapq import heappop, heappush

def read_input():
    N, Q = map(int, sys.stdin.readline().split())
    edges = [[] for _ in range(N+Q+1)]
    for i in range(Q):
        L, R, C = map(int, sys.stdin.readline().split())
        for j in range(L, R+1):
            edges[N+i+1].append((C, j))
            edges[j].append((C, N+i+1))
    return N, Q, edges

def minimum_spanning_tree(N, Q, edges):
    visited = [False] * (N+Q+1)
    heap = [(0, 1)]
    total_cost = 0
    while heap:
        cost, node = heappop(heap)
        if not visited[node]:
            visited[node] = True
            total_cost += cost
            for next_cost, next_node in edges[node]:
                if not visited[next_node]:
                    heappush(heap, (next_cost, next_node))
    return total_cost if all(visited) else -1

N, Q, edges = read_input()
print(minimum_spanning_tree(N, Q, edges))