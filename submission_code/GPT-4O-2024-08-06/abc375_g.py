# YOUR CODE HERE
import sys
import heapq

def dijkstra(n, graph, start):
    dist = [float('inf')] * (n + 1)
    dist[start] = 0
    pq = [(0, start)]  # (distance, node)
    
    while pq:
        current_dist, u = heapq.heappop(pq)
        
        if current_dist > dist[u]:
            continue
        
        for v, length in graph[u]:
            if dist[u] + length < dist[v]:
                dist[v] = dist[u] + length
                heapq.heappush(pq, (dist[v], v))
    
    return dist

def solve():
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    N = int(data[index])
    index += 1
    M = int(data[index])
    index += 1
    
    edges = []
    graph = [[] for _ in range(N + 1)]
    
    for i in range(M):
        A = int(data[index])
        index += 1
        B = int(data[index])
        index += 1
        C = int(data[index])
        index += 1
        edges.append((A, B, C))
        graph[A].append((B, C))
        graph[B].append((A, C))
    
    # Find the shortest path from 1 to N with all roads
    full_dist = dijkstra(N, graph, 1)
    full_shortest_path = full_dist[N]
    
    results = []
    
    for i in range(M):
        A, B, C = edges[i]
        
        # Create a new graph without the i-th road
        new_graph = [[] for _ in range(N + 1)]
        for j in range(M):
            if j == i:
                continue
            a, b, c = edges[j]
            new_graph[a].append((b, c))
            new_graph[b].append((a, c))
        
        # Find the shortest path from 1 to N without the i-th road
        new_dist = dijkstra(N, new_graph, 1)
        new_shortest_path = new_dist[N]
        
        # Compare the distances
        if new_shortest_path != full_shortest_path:
            results.append("Yes")
        else:
            results.append("No")
    
    # Output results
    for result in results:
        print(result)