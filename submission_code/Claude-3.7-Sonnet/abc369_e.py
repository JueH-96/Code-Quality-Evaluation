import heapq
from itertools import permutations

def shortest_paths(graph, source, n):
    # Dijkstra's algorithm to compute shortest paths from source to all other nodes
    dist = [float('inf')] * (n + 1)
    dist[source] = 0
    pq = [(0, source)]
    
    while pq:
        d, node = heapq.heappop(pq)
        if d > dist[node]:
            continue
        
        for neighbor, weight in graph[node]:
            if dist[neighbor] > d + weight:
                dist[neighbor] = d + weight
                heapq.heappush(pq, (d + weight, neighbor))
    
    return dist

def main():
    N, M = map(int, input().split())
    
    bridges = []
    for _ in range(M):
        u, v, t = map(int, input().split())
        bridges.append((u, v, t))
    
    Q = int(input())
    
    query_bridges = []
    for _ in range(Q):
        K = int(input())
        bridge_indices = list(map(int, input().split()))
        query_bridges.append(bridge_indices)
    
    # Create an adjacency list representation of the graph
    graph = [[] for _ in range(N + 1)]
    
    for i, (u, v, t) in enumerate(bridges):
        graph[u].append((v, t))
        graph[v].append((u, t))  # Bidirectional
    
    # Compute shortest paths from each island to every other island
    dist = {}
    for source in range(1, N + 1):
        dist[source] = shortest_paths(graph, source, N)
    
    for bridges_to_include in query_bridges:
        min_time = float('inf')
        
        # Extract bridge details
        bridge_details = [(bridges[idx-1][0], bridges[idx-1][1], bridges[idx-1][2]) for idx in bridges_to_include]
        k = len(bridges_to_include)
        
        # Try all possible orderings of bridges
        for p in permutations(range(k)):
            # Try all possible orientations of bridges
            for mask in range(1 << k):
                total_time = 0
                current_island = 1  # Start from island 1
                
                for i in range(k):
                    bridge_pos = p[i]
                    u, v, t = bridge_details[bridge_pos]
                    
                    # Determine bridge orientation
                    if (mask >> bridge_pos) & 1:
                        start, end = v, u
                    else:
                        start, end = u, v
                    
                    # Add time to reach the starting island of the bridge
                    total_time += dist[current_island][start]
                    
                    # Cross the bridge
                    total_time += t
                    
                    current_island = end
                
                # Add time to reach island N from the last bridge
                total_time += dist[current_island][N]
                
                min_time = min(min_time, total_time)
        
        print(min_time)

if __name__ == "__main__":
    main()