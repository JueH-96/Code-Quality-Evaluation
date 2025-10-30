from collections import defaultdict, deque

def build_graph(N, edges):
    graph = defaultdict(list)
    for a, b in edges:
        graph[a].append(b)
        graph[b].append(a)
    return graph

def find_path(graph, start, end, parent):
    visited = set()
    q = deque([start])
    visited.add(start)
    
    while q:
        curr = q.popleft()
        if curr == end:
            return True
            
        for next_node in graph[curr]:
            if next_node not in visited:
                visited.add(next_node)
                parent[next_node] = curr
                q.append(next_node)
    return False

def get_path(parent, start, end):
    path = []
    curr = end
    while curr != start:
        path.append(curr)
        curr = parent[curr]
    path.append(start)
    return set(path)

def solve():
    N, K = map(int, input().split())
    edges = []
    for _ in range(N-1):
        a, b = map(int, input().split())
        edges.append((a, b))
    vertices = list(map(int, input().split()))
    
    if K == 1:
        return 1
        
    graph = build_graph(N, edges)
    required_vertices = set()
    
    for i in range(K-1):
        parent = defaultdict(int)
        find_path(graph, vertices[i], vertices[i+1], parent)
        path = get_path(parent, vertices[i], vertices[i+1])
        required_vertices.update(path)
    
    return len(required_vertices)

print(solve())