from collections import defaultdict

def read_input():
    N = int(input())
    edges = []
    for _ in range(N-1):
        a, b = map(int, input().split())
        edges.append((a, b))
    return N, edges

def build_graph(N, edges):
    graph = defaultdict(list)
    for a, b in edges:
        graph[a].append(b)
        graph[b].append(a)
    return graph

def get_leaves(graph):
    leaves = []
    for v in graph:
        if len(graph[v]) == 1:
            leaves.append(v)
    return leaves

def find_path(graph, start, end, visited=None, path=None):
    if visited is None:
        visited = set()
    if path is None:
        path = []
        
    visited.add(start)
    path.append(start)
    
    if start == end:
        return path
        
    for next_v in graph[start]:
        if next_v not in visited:
            result = find_path(graph, next_v, end, visited, path)
            if result:
                return result
                
    path.pop()
    return None

def path_length(graph, u, v):
    path = find_path(graph, u, v)
    return len(path) - 1

def remove_vertices(graph, u, v):
    # Remove edges connected to u
    for neighbor in graph[u]:
        graph[neighbor].remove(u)
    # Remove edges connected to v
    for neighbor in graph[v]:
        graph[neighbor].remove(v)
    # Remove vertices
    del graph[u]
    del graph[v]

def solve():
    N, edges = read_input()
    graph = build_graph(N, edges)
    result = []
    
    for _ in range(N//2):
        leaves = get_leaves(graph)
        max_dist = -1
        best_pair = None
        
        # Try all pairs of leaves
        for i in range(len(leaves)):
            for j in range(i+1, len(leaves)):
                u, v = leaves[i], leaves[j]
                dist = path_length(graph, u, v)
                if dist > max_dist:
                    max_dist = dist
                    best_pair = (u, v)
        
        u, v = best_pair
        result.append((u, v))
        remove_vertices(graph, u, v)
    
    for u, v in result:
        print(u, v)

solve()