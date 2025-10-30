import sys
import heapq
from collections import defaultdict

def dijkstra(graph, start, n):
    distances = {i: float('inf') for i in range(1, n + 1)}
    distances[start] = 0
    priority_queue = [(0, start)]
    
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        
        if current_distance > distances[current_node]:
            continue
        
        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
    
    return distances

def solve():
    input = sys.stdin.read
    data = input().splitlines()
    
    idx = 0
    N, M = map(int, data[idx].split())
    idx += 1
    
    graph = defaultdict(list)
    bridges = []
    
    for i in range(M):
        U, V, T = map(int, data[idx].split())
        graph[U].append((V, T))
        graph[V].append((U, T))
        bridges.append((U, V, T))
        idx += 1
    
    Q = int(data[idx])
    idx += 1
    
    results = []
    
    for _ in range(Q):
        K = int(data[idx])
        idx += 1
        bridge_indices = list(map(int, data[idx].split()))
        idx += 1
        
        # Create a subgraph using the specified bridges
        subgraph = defaultdict(list)
        total_time = 0
        
        for b in bridge_indices:
            U, V, T = bridges[b - 1]
            subgraph[U].append((V, T))
            subgraph[V].append((U, T))
            total_time += T
        
        # Find the shortest path from island 1 to island N
        distances = dijkstra(subgraph, 1, N)
        min_time = distances[N]
        
        if min_time == float('inf'):
            results.append(-1)  # In case there's no path, but per problem statement, this shouldn't happen
        else:
            results.append(min_time + total_time)
    
    print("
".join(map(str, results)))