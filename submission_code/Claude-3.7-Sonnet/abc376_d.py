from collections import deque

def shortest_cycle_with_vertex_1(graph, n):
    # Start BFS from vertex 1
    queue = deque([1])
    distance = [float('inf')] * (n+1)
    distance[1] = 0
    
    # Find shortest path from vertex 1 to all other vertices
    while queue:
        vertex = queue.popleft()
        
        for next_vertex in graph[vertex]:
            if distance[next_vertex] == float('inf'):
                distance[next_vertex] = distance[vertex] + 1
                queue.append(next_vertex)
    
    # Find shortest cycle containing vertex 1
    min_cycle_length = float('inf')
    for vertex in range(1, n+1):
        for next_vertex in graph[vertex]:
            if next_vertex == 1 and distance[vertex] != float('inf'):
                cycle_length = distance[vertex] + 1
                min_cycle_length = min(min_cycle_length, cycle_length)
    
    if min_cycle_length == float('inf'):
        return -1  # No cycle found
    return min_cycle_length

# Read input
n, m = map(int, input().split())
graph = [[] for _ in range(n+1)]  # 1-indexed graph

for _ in range(m):
    a, b = map(int, input().split())
    graph[a].append(b)

# Find the shortest cycle
result = shortest_cycle_with_vertex_1(graph, n)
print(result)