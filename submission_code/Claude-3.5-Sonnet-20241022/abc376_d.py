from collections import deque
import sys

def find_min_cycle(N, edges):
    # Create adjacency list representation
    graph = [[] for _ in range(N+1)]
    for a, b in edges:
        graph[a].append(b)
    
    # BFS from vertex 1 to find shortest cycle
    def bfs_shortest_cycle():
        # Queue will store (vertex, distance, path)
        queue = deque([(1, 0, [1])])
        # Keep track of visited vertices and their distances
        visited = {1: 0}
        
        while queue:
            vertex, dist, path = queue.popleft()
            
            # Check all neighbors
            for next_vertex in graph[vertex]:
                # If we found vertex 1 and it's not the immediate previous vertex
                if next_vertex == 1 and len(path) > 1:
                    return dist + 1
                
                # If vertex not visited or we found a shorter path
                if next_vertex not in visited or visited[next_vertex] > dist + 1:
                    visited[next_vertex] = dist + 1
                    new_path = path + [next_vertex]
                    queue.append((next_vertex, dist + 1, new_path))
        
        return float('inf')
    
    min_cycle = bfs_shortest_cycle()
    return min_cycle if min_cycle != float('inf') else -1

# Read input
N, M = map(int, input().split())
edges = []
for _ in range(M):
    a, b = map(int, input().split())
    edges.append((a, b))

# Find and print result
result = find_min_cycle(N, edges)
print(result)