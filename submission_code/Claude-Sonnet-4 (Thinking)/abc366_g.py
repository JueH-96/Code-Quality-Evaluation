def solve():
    N, M = map(int, input().split())
    
    # Build adjacency list
    adj = [[] for _ in range(N)]
    for _ in range(M):
        u, v = map(int, input().split())
        u -= 1  # Convert to 0-indexed
        v -= 1
        adj[u].append(v)
        adj[v].append(u)
    
    # Find connected components
    visited = [False] * N
    components = []
    
    def dfs(v, component):
        visited[v] = True
        component.append(v)
        for u in adj[v]:
            if not visited[u]:
                dfs(u, component)
    
    for i in range(N):
        if not visited[i]:
            component = []
            dfs(i, component)
            components.append(component)
    
    # Check if solution exists
    degrees = [len(adj[i]) for i in range(N)]
    
    for component in components:
        if len(component) > 1:
            # Check if any vertex has degree 1
            for v in component:
                if degrees[v] == 1:
                    print("No")
                    return
    
    # Construct solution
    result = [1] * N
    
    for component in components:
        if len(component) == 1:
            # Isolated vertex
            result[component[0]] = 1
        else:
            # Connected component where every vertex has degree >= 2
            # Check if all vertices have even degree
            all_even = all(degrees[v] % 2 == 0 for v in component)
            
            if all_even:
                # Assign same value to all vertices
                for v in component:
                    result[v] = 4
            else:
                # Need to find a valid assignment
                # Try different values until constraints are satisfied
                found = False
                
                # Try assigning same value to all (works for odd cycles)
                for val in [1, 2, 3, 4, 5]:
                    for v in component:
                        result[v] = val
                    
                    # Check constraints
                    valid = True
                    for v in component:
                        xor_sum = 0
                        for u in adj[v]:
                            xor_sum ^= result[u]
                        if xor_sum != 0:
                            valid = False
                            break
                    
                    if valid:
                        found = True
                        break
                
                if not found:
                    # Try alternating pattern for bipartite-like structures
                    # Color the component with 2 colors
                    color = [-1] * N
                    queue = [component[0]]
                    color[component[0]] = 0
                    
                    while queue:
                        v = queue.pop(0)
                        for u in adj[v]:
                            if u in component:
                                if color[u] == -1:
                                    color[u] = 1 - color[v]
                                    queue.append(u)
                    
                    # Assign values based on color
                    for v in component:
                        result[v] = 4 if color[v] == 0 else 8
                    
                    # Verify this works
                    valid = True
                    for v in component:
                        xor_sum = 0
                        for u in adj[v]:
                            xor_sum ^= result[u]
                        if xor_sum != 0:
                            # If bipartite coloring doesn't work, use same value for all
                            for v2 in component:
                                result[v2] = 4
                            break
    
    print("Yes")
    print(' '.join(map(str, result)))

solve()