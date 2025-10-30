# YOUR CODE HERE
import sys
from collections import deque

def bfs_distance(start, graph, n):
    # BFS to find shortest path distances from start
    distances = [-1] * (n + 1)
    queue = deque([start])
    distances[start] = 0
    
    while queue:
        node = queue.popleft()
        current_distance = distances[node]
        
        for neighbor in graph[node]:
            if distances[neighbor] == -1:  # Not visited
                distances[neighbor] = current_distance + 1
                queue.append(neighbor)
    
    return distances

def main():
    input = sys.stdin.read
    data = input().split()
    
    N1 = int(data[0])
    N2 = int(data[1])
    M = int(data[2])
    
    edges = []
    index = 3
    for _ in range(M):
        a = int(data[index])
        b = int(data[index + 1])
        edges.append((a, b))
        index += 2
    
    # Create adjacency list for the graph
    graph = [[] for _ in range(N1 + N2 + 1)]
    
    for a, b in edges:
        graph[a].append(b)
        graph[b].append(a)
    
    # Calculate distances from vertex 1
    dist_from_1 = bfs_distance(1, graph, N1 + N2)
    
    # Calculate distances from vertex N1 + N2
    dist_from_N1_N2 = bfs_distance(N1 + N2, graph, N1 + N2)
    
    # Find the maximum possible d
    max_d = 0
    for u in range(1, N1 + 1):
        for v in range(N1 + 1, N1 + N2 + 1):
            # Calculate the path length if we add an edge between u and v
            if dist_from_1[u] != -1 and dist_from_N1_N2[v] != -1:
                d = dist_from_1[u] + 1 + dist_from_N1_N2[v]
                max_d = max(max_d, d)
    
    print(max_d)