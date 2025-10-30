import sys
import heapq
from collections import defaultdict

def dijkstra(start, end, graph, required_edges):
    # Priority queue for Dijkstra's algorithm
    pq = []
    # Distance table
    dist = {node: float('inf') for node in graph}
    dist[start] = 0
    heapq.heappush(pq, (0, start))
    
    while pq:
        current_dist, current_node = heapq.heappop(pq)
        
        if current_dist > dist[current_node]:
            continue
        
        for neighbor, time in graph[current_node]:
            if (current_node, neighbor) in required_edges or (neighbor, current_node) in required_edges:
                distance = current_dist + time
                if distance < dist[neighbor]:
                    dist[neighbor] = distance
                    heapq.heappush(pq, (distance, neighbor))
    
    return dist[end]

def main():
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    N = int(data[index])
    index += 1
    M = int(data[index])
    index += 1
    
    bridges = []
    for _ in range(M):
        U = int(data[index])
        index += 1
        V = int(data[index])
        index += 1
        T = int(data[index])
        index += 1
        bridges.append((U, V, T))
    
    Q = int(data[index])
    index += 1
    
    queries = []
    for _ in range(Q):
        K = int(data[index])
        index += 1
        B = []
        for _ in range(K):
            B.append(int(data[index]) - 1)  # Convert to zero-based index
            index += 1
        queries.append(B)
    
    # Process each query
    results = []
    for query in queries:
        # Create a graph with only the required bridges
        graph = defaultdict(list)
        required_edges = set()
        
        for bridge_index in query:
            U, V, T = bridges[bridge_index]
            graph[U].append((V, T))
            graph[V].append((U, T))
            required_edges.add((U, V))
            required_edges.add((V, U))
        
        # Find the shortest path from 1 to N using Dijkstra's algorithm
        min_time = dijkstra(1, N, graph, required_edges)
        results.append(min_time)
    
    for result in results:
        print(result)