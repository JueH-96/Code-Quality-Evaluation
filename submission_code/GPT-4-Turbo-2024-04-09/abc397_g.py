import sys
import heapq
from itertools import combinations

input = sys.stdin.read
def main():
    data = input().split()
    index = 0
    N = int(data[index])
    index += 1
    M = int(data[index])
    index += 1
    K = int(data[index])
    index += 1
    
    edges = []
    for _ in range(M):
        u = int(data[index])
        index += 1
        v = int(data[index])
        index += 1
        edges.append((u, v))
    
    def shortest_path(weighted_edges):
        # Using Dijkstra's algorithm to find the shortest path from 1 to N
        dist = [float('inf')] * (N + 1)
        dist[1] = 0
        priority_queue = [(0, 1)]  # (distance, vertex)
        
        while priority_queue:
            d, u = heapq.heappop(priority_queue)
            if d > dist[u]:
                continue
            for cost, v in weighted_edges[u]:
                if dist[u] + cost < dist[v]:
                    dist[v] = dist[u] + cost
                    heapq.heappush(priority_queue, (dist[v], v))
        
        return dist[N]
    
    # Generate all possible combinations of edges to be weighted as 1
    max_distance = 0
    for edges_to_weight in combinations(range(M), K):
        # Create adjacency list with weights
        weighted_edges = [[] for _ in range(N + 1)]
        for i, (u, v) in enumerate(edges):
            if i in edges_to_weight:
                weighted_edges[u].append((1, v))
            else:
                weighted_edges[u].append((0, v))
        
        # Calculate the shortest path with the current weighting
        distance = shortest_path(weighted_edges)
        max_distance = max(max_distance, distance)
    
    print(max_distance)

if __name__ == "__main__":
    main()