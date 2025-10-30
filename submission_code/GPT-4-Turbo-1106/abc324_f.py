import sys
from collections import defaultdict
from heapq import heappop, heappush

def dijkstra(graph, start, N):
    # Initialize distances and priority queue
    distances = [float('inf')] * (N + 1)
    distances[start] = 0
    pq = [(0, start, 0, 0)]  # (distance, node, beauty, cost)

    while pq:
        dist, node, beauty, cost = heappop(pq)
        if node == N:
            return beauty / cost if cost != 0 else 0
        for next_node, b, c in graph[node]:
            new_dist = dist + c
            if new_dist < distances[next_node]:
                distances[next_node] = new_dist
                heappush(pq, (new_dist, next_node, beauty + b, cost + c))
    return 0

def solve(N, M, edges):
    graph = defaultdict(list)
    for u, v, b, c in edges:
        graph[u].append((v, b, c))

    # Find the maximum beauty/cost ratio using Dijkstra's algorithm
    max_ratio = dijkstra(graph, 1, N)
    return max_ratio

def main():
    input_data = sys.stdin.read().splitlines()
    N, M = map(int, input_data[0].split())
    edges = [tuple(map(int, line.split())) for line in input_data[1:]]
    max_ratio = solve(N, M, edges)
    print(f"{max_ratio:.9f}")

if __name__ == "__main__":
    main()