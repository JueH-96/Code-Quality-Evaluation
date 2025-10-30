def find_all_paths(graph, start, end, path=None, visited=None):
    if path is None:
        path = []
    if visited is None:
        visited = set()
        
    path = path + [start]
    visited.add(start)
    
    if start == end:
        yield path
    else:
        for neighbor, _ in graph[start]:
            if neighbor not in visited:
                yield from find_all_paths(graph, neighbor, end, path, visited.copy())

def calculate_path_xor(path, graph):
    xor_result = 0
    for i in range(len(path)-1):
        current = path[i]
        next_vertex = path[i+1]
        # Find the edge weight between current and next_vertex
        for neighbor, weight in graph[current]:
            if neighbor == next_vertex:
                xor_result ^= weight
                break
    return xor_result

def solve():
    # Read input
    N, M = map(int, input().split())
    
    # Create adjacency list representation of the graph
    graph = [[] for _ in range(N+1)]
    
    # Read edges
    for _ in range(M):
        u, v, w = map(int, input().split())
        # Add edges in both directions since it's undirected
        graph[u].append((v, w))
        graph[v].append((u, w))
    
    # Find all simple paths from 1 to N
    all_paths = list(find_all_paths(graph, 1, N))
    
    # Calculate XOR for each path and find minimum
    min_xor = float('inf')
    for path in all_paths:
        path_xor = calculate_path_xor(path, graph)
        min_xor = min(min_xor, path_xor)
    
    print(min_xor)

# Run the solution
solve()