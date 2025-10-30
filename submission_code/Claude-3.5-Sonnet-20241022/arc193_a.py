from collections import defaultdict, deque

def has_intersection(l1, r1, l2, r2):
    # Returns True if intervals [l1,r1] and [l2,r2] have no intersection
    return r1 < l2 or r2 < l1

def build_graph(N, L, R):
    # Build adjacency list representation of the graph
    graph = defaultdict(list)
    for i in range(N):
        for j in range(i+1, N):
            # Add edge if intervals have no intersection
            if has_intersection(L[i], R[i], L[j], R[j]):
                graph[i+1].append(j+1)
                graph[j+1].append(i+1)
    return graph

def find_min_weight_path(graph, start, end, weights, N):
    # Use BFS to find shortest path with minimum weight
    if start not in graph or end not in graph:
        return -1
        
    visited = set()
    queue = deque([(start, weights[start-1])])  # (vertex, path_weight)
    visited.add(start)
    parent = {start: None}
    
    while queue:
        vertex, weight = queue.popleft()
        
        if vertex == end:
            return weight
            
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = vertex
                queue.append((neighbor, weight + weights[neighbor-1]))
                
    return -1

# Read input
N = int(input())
W = list(map(int, input().split()))
L = []
R = []
for _ in range(N):
    l, r = map(int, input().split())
    L.append(l)
    R.append(r)

# Build graph
graph = build_graph(N, L, R)

# Process queries
Q = int(input())
for _ in range(Q):
    s, t = map(int, input().split())
    # Find minimum weight path
    result = find_min_weight_path(graph, s, t, W, N)
    print(result)