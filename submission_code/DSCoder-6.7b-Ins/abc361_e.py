import sys
from heapq import heappop, heappush

def read_input():
    N = int(sys.stdin.readline().strip())
    edges = [[] for _ in range(N+1)]
    for _ in range(N-1):
        A, B, C = map(int, sys.stdin.readline().strip().split())
        edges[A].append((B, C))
        edges[B].append((A, C))
    return N, edges

def solve():
    N, edges = read_input()
    visited = [False] * (N+1)
    heap = [(0, 1)]  # start from city 1
    total_distance = 0
    while heap:
        d, node = heappop(heap)
        if not visited[node]:
            visited[node] = True
            total_distance += d
            for neighbor, distance in edges[node]:
                if not visited[neighbor]:
                    heappush(heap, (distance, neighbor))
    return total_distance

print(solve())