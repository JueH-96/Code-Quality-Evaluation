import sys
import heapq
from collections import defaultdict

def dijkstra(n, graph, start):
    distances = [float('inf')] * (n + 1)
    distances[start] = 0
    priority_queue = [(0, start)]  # (distance, node)
    
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

def main():
    input = sys.stdin.read
    data = input().splitlines()
    
    N, M = map(int, data[0].split())
    edges = []
    graph = defaultdict(list)
    
    for i in range(1, M + 1):
        A, B, C = map(int, data[i].split())
        edges.append((A, B, C))
        graph[A].append((B, C))
        graph[B].append((A, C))
    
    # Calculate the shortest distance from city 1 to city N with all roads
    all_distances = dijkstra(N, graph, 1)
    shortest_with_all = all_distances[N]
    
    results = []
    
    for i in range(M):
        A, B, C = edges[i]
        
        # Create a new graph without the i-th edge
        new_graph = defaultdict(list)
        for j in range(M):
            if j != i:
                A_j, B_j, C_j = edges[j]
                new_graph[A_j].append((B_j, C_j))
                new_graph[B_j].append((A_j, C_j))
        
        # Calculate the shortest distance from city 1 to city N without the i-th edge
        new_distances = dijkstra(N, new_graph, 1)
        shortest_without_i = new_distances[N]
        
        # Check if the distances are different
        if (shortest_with_all != shortest_without_i) or (shortest_with_all == float('inf') and shortest_without_i != float('inf')):
            results.append("Yes")
        else:
            results.append("No")
    
    # Print the results
    sys.stdout.write("
".join(results) + "
")

if __name__ == "__main__":
    main()