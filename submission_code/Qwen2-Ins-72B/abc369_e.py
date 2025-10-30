from heapq import heappush, heappop
from collections import defaultdict

def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    priority_queue = [(0, start)]
    
    while priority_queue:
        current_distance, current_node = heappop(priority_queue)
        
        if current_distance > distances[current_node]:
            continue
        
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heappush(priority_queue, (distance, neighbor))
    
    return distances

def solve(N, M, bridges, Q, queries):
    # Build the graph
    graph = defaultdict(lambda: defaultdict(int))
    for i, (u, v, t) in enumerate(bridges, start=1):
        graph[u][v] = t
        graph[v][u] = t
    
    for _ in range(Q):
        K, *B = map(int, input().split())
        
        # Remove the bridges specified in the query
        for b in B:
            u, v, t = bridges[b-1]
            graph[u].pop(v)
            graph[v].pop(u)
        
        # Calculate the shortest path from island 1 to island N
        distances = dijkstra(graph, 1)
        min_time = distances[N]
        
        # Restore the removed bridges
        for b in B:
            u, v, t = bridges[b-1]
            graph[u][v] = t
            graph[v][u] = t
        
        print(min_time)

# Read the input
N, M = map(int, input().split())
bridges = [tuple(map(int, input().split())) for _ in range(M)]
Q = int(input())

# Solve the problem
solve(N, M, bridges, Q)