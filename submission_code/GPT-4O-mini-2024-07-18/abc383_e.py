import sys
import heapq
from collections import defaultdict

def dijkstra_max_weight(graph, start, n):
    # Initialize distances with infinity
    max_weight = [float('inf')] * (n + 1)
    max_weight[start] = 0
    priority_queue = [(0, start)]  # (max_edge_weight, vertex)

    while priority_queue:
        current_weight, current_vertex = heapq.heappop(priority_queue)

        if current_weight > max_weight[current_vertex]:
            continue

        for neighbor, weight in graph[current_vertex]:
            new_weight = max(current_weight, weight)
            if new_weight < max_weight[neighbor]:
                max_weight[neighbor] = new_weight
                heapq.heappush(priority_queue, (new_weight, neighbor))

    return max_weight

def main():
    input = sys.stdin.read
    data = input().splitlines()
    
    # Read N, M, K
    N, M, K = map(int, data[0].split())
    
    # Read edges
    graph = defaultdict(list)
    for i in range(1, M + 1):
        u, v, w = map(int, data[i].split())
        graph[u].append((v, w))
        graph[v].append((u, w))
    
    # Read sequences A and B
    A = list(map(int, data[M + 1].split()))
    B = list(map(int, data[M + 2].split()))
    
    # Calculate the maximum edge weight from each A_i to all other vertices
    max_weights = {}
    for a in A:
        max_weights[a] = dijkstra_max_weight(graph, a, N)

    # Sort B to minimize the sum
    B.sort()
    
    # Calculate the minimum possible sum
    min_sum = 0
    for i in range(K):
        min_sum += max_weights[A[i]][B[i]]

    print(min_sum)

if __name__ == "__main__":
    main()