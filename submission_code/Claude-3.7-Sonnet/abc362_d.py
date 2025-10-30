import heapq

def main():
    # Parse input
    N, M = map(int, input().split())
    vertex_weights = [0] + list(map(int, input().split()))  # Make 1-indexed
    
    # Build the graph as adjacency list
    graph = [[] for _ in range(N+1)]
    for _ in range(M):
        u, v, w = map(int, input().split())
        graph[u].append((v, w))
        graph[v].append((u, w))  # Undirected graph
    
    # Run Dijkstra's algorithm
    # Initialize distances
    distances = [float('inf')] * (N+1)
    distances[1] = vertex_weights[1]  # Start with the weight of vertex 1
    
    # Priority queue for Dijkstra
    pq = [(distances[1], 1)]
    
    while pq:
        dist, node = heapq.heappop(pq)
        
        if dist > distances[node]:
            continue
        
        for neighbor, edge_weight in graph[node]:
            # New distance includes the current distance, edge weight, and neighbor vertex weight
            new_dist = dist + edge_weight + vertex_weights[neighbor]
            
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))
    
    # Output results
    print(" ".join(map(str, distances[2:])))

if __name__ == "__main__":
    main()