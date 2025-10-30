from collections import defaultdict, deque

def build_graph(N, edges):
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    return graph

def get_degree(graph):
    return {v: len(neighbors) for v, neighbors in graph.items()}

def is_leaf(graph, vertex, degree):
    return degree[vertex] == 1

def find_center_candidates(N, graph):
    degree = get_degree(graph)
    queue = deque()
    
    # Add leaves to queue
    for v in range(1, N+1):
        if is_leaf(graph, v, degree):
            queue.append(v)
    
    remaining = N
    while remaining > 2:
        if not queue:
            break
        vertex = queue.popleft()
        remaining -= 1
        
        # Update neighbor degrees
        for neighbor in graph[vertex]:
            degree[neighbor] -= 1
            if degree[neighbor] == 1:
                queue.append(neighbor)
    
    # Return remaining vertices (center candidates)
    return [v for v in range(1, N+1) if degree[v] > 0]

def check_snowflake(center, graph, N):
    visited = set()
    queue = deque([(center, 0)])
    visited.add(center)
    levels = defaultdict(set)
    
    while queue:
        vertex, level = queue.popleft()
        levels[level].add(vertex)
        
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, level + 1))
    
    if len(levels) > 3:
        return N  # Not possible
    
    if len(levels) <= 2:
        return N - len(visited)
    
    # Check if it forms valid snowflake tree
    center_degree = len(graph[center])
    middle_vertices = levels[1]
    leaves = levels[2]
    
    # All middle vertices should have same number of leaves
    leaf_counts = defaultdict(int)
    for middle in middle_vertices:
        leaf_count = sum(1 for neighbor in graph[middle] if neighbor in leaves)
        leaf_counts[leaf_count] += 1
    
    if len(leaf_counts) != 1:
        return N - len(visited)
    
    return N - len(visited)

def solve():
    # Read input
    N = int(input())
    edges = []
    for _ in range(N-1):
        u, v = map(int, input().split())
        edges.append((u, v))
    
    # Build graph
    graph = build_graph(N, edges)
    
    # Find potential centers
    centers = find_center_candidates(N, graph)
    
    # Try each center and find minimum deletions needed
    min_deletions = N
    for center in centers:
        deletions = check_snowflake(center, graph, N)
        min_deletions = min(min_deletions, deletions)
    
    print(min_deletions)

solve()