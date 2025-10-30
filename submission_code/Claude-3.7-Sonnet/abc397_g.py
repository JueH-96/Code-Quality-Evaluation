from collections import deque
from itertools import combinations

def zero_one_bfs(graph, start, end, edges_with_weight_1):
    """
    Perform a 0-1 BFS to find the shortest path from start to end
    with the given edges having weight 1 and all others having weight 0.
    """
    distances = [float('inf')] * (len(graph) + 1)
    distances[start] = 0
    queue = deque([start])
    
    while queue:
        current_node = queue.popleft()
        
        # If we've reached the target, we can return the distance
        if current_node == end:
            return distances[end]
        
        for edge_idx, neighbor in graph[current_node]:
            weight = 1 if edge_idx in edges_with_weight_1 else 0
            distance = distances[current_node] + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                if weight == 0:
                    queue.appendleft(neighbor)  # Process 0-weight edges first
                else:
                    queue.append(neighbor)  # Process 1-weight edges later
    
    return distances[end]

def max_shortest_distance(N, M, K, edges):
    """
    Find the maximum possible shortest distance from vertex 1 to vertex N
    after changing exactly K edges to have weight 1.
    """
    # Build the graph representation
    graph = [[] for _ in range(N + 1)]
    for i, (u, v) in enumerate(edges):
        graph[u].append((i, v))  # Store edge index along with the neighbor
    
    max_distance = 0
    
    # Try all combinations of K edges to set to weight 1
    for selected_edges in combinations(range(M), K):
        # Compute the shortest path with these edges having weight 1
        distance = zero_one_bfs(graph, 1, N, set(selected_edges))
        max_distance = max(max_distance, distance)
    
    return max_distance

# Read input
N, M, K = map(int, input().split())
edges = []
for _ in range(M):
    u, v = map(int, input().split())
    edges.append((u, v))

# Print the maximum possible shortest distance
print(max_shortest_distance(N, M, K, edges))