import collections

def solve():
    n, m = map(int, input().split())
    initial_edges = []
    for _ in range(m):
        u, v = map(int, input().split())
        initial_edges.append((u, v))
    
    adj = collections.defaultdict(list)
    for u, v in initial_edges:
        adj[u].append(v)
        adj[v].append(u)
        
    visited = [False] * (n + 1)
    components = []
    for i in range(1, n + 1):
        if not visited[i]:
            component = []
            q = collections.deque([i])
            visited[i] = True
            component.append(i)
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        component.append(v)
                        q.append(v)
            components.append(component)
            
    total_operations = 0
    for component_vertices in components:
        num_vertices = len(component_vertices)
        initial_component_edges = 0
        for u1, v1 in initial_edges:
            if u1 in component_vertices and v1 in component_vertices:
                initial_component_edges += 1
        
        possible_edges_in_complete_graph = num_vertices * (num_vertices - 1) // 2
        operations_for_component = possible_edges_in_complete_graph - initial_component_edges
        total_operations += operations_for_component
        
    print(total_operations)

if __name__ == '__main__':
    solve()