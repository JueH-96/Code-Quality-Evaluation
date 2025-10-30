# YOUR CODE HERE
import sys
from collections import defaultdict
from heapq import heappush, heappop

def read_input():
    N, M = map(int, input().split())
    graph = defaultdict(list)
    for _ in range(M):
        u, v, b, c = map(int, input().split())
        graph[u].append((v, b, c))
    return N, graph

def parametric_search(N, graph):
    def dijkstra(x):
        dist = [float('inf')] * (N + 1)
        dist[1] = 0
        pq = [(0, 1)]
        
        while pq:
            d, u = heappop(pq)
            if u == N:
                return True
            if d > dist[u]:
                continue
            for v, b, c in graph[u]:
                new_d = d + c - x * b
                if new_d < dist[v]:
                    dist[v] = new_d
                    heappush(pq, (new_d, v))
        return False

    left, right = 0, 10**4
    for _ in range(100):  # Binary search iterations
        mid = (left + right) / 2
        if dijkstra(mid):
            left = mid
        else:
            right = mid
    return left

def main():
    N, graph = read_input()
    result = parametric_search(N, graph)
    print(f"{result:.10f}")

if __name__ == "__main__":
    main()