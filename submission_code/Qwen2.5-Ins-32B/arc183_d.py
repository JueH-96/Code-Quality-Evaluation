# YOUR CODE HERE
import sys
from collections import deque

def read_ints(): return map(int, sys.stdin.readline().strip().split())

def bfs(graph, start):
    queue = deque([start])
    visited = {start}
    parent = {start: None}
    distance = {start: 0}
    
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = node
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)
    
    return parent, distance

def find_farthest_leaf(graph, start):
    _, distance = bfs(graph, start)
    farthest_leaf = max(distance, key=distance.get)
    return farthest_leaf, distance[farthest_leaf]

def find_path(parent, start, end):
    path = []
    while end != start:
        path.append(end)
        end = parent[end]
    path.append(start)
    return path[::-1]

def solve():
    N = int(input())
    edges = [tuple(read_ints()) for _ in range(N-1)]
    
    graph = {i: set() for i in range(1, N+1)}
    for u, v in edges:
        graph[u].add(v)
        graph[v].add(u)
    
    operations = []
    for _ in range(N // 2):
        leaves = [node for node in graph if len(graph[node]) == 1]
        start = leaves[0]
        farthest_leaf, _ = find_farthest_leaf(graph, start)
        _, distance = bfs(graph, farthest_leaf)
        farthest_leaf = max(distance, key=distance.get)
        
        parent, _ = bfs(graph, farthest_leaf)
        path = find_path(parent, start, farthest_leaf)
        
        operations.append((path[0], path[-1]))
        
        for node in path:
            for neighbor in list(graph[node]):
                graph[neighbor].remove(node)
                if len(graph[neighbor]) == 0:
                    del graph[neighbor]
            del graph[node]
    
    for u, v in operations:
        print(u, v)

if __name__ == "__main__":
    solve()