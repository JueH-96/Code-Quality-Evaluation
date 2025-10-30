def solve():
    # Read input
    N, M = map(int, input().split())
    A = list(map(int, input().split()))
    MOD = 998244353
    
    # Create a graph representation of dependencies
    graph = [[] for _ in range(N)]
    for i in range(N):
        graph[i].append(A[i] - 1)  # Convert to 0-based indexing
    
    # Find strongly connected components using Kosaraju's algorithm
    def dfs1(v, visited, order):
        visited[v] = True
        for u in graph[v]:
            if not visited[u]:
                dfs1(u, visited, order)
        order.append(v)
    
    def dfs2(v, visited, component):
        visited[v] = True
        component.append(v)
        for u in graph[v]:
            if not visited[u]:
                dfs2(u, visited, component)
    
    # First DFS pass
    visited = [False] * N
    order = []
    for i in range(N):
        if not visited[i]:
            dfs1(i, visited, order)
    
    # Find SCCs
    visited = [False] * N
    components = []
    for v in reversed(order):
        if not visited[v]:
            component = []
            dfs2(v, visited, component)
            components.append(component)
    
    # For each component, find the maximum possible value
    max_val = [0] * N
    for component in components:
        max_component = M
        for v in component:
            for u in graph[v]:
                if u not in component:
                    max_component = min(max_component, max_val[u])
        for v in component:
            max_val[v] = max_component
    
    # Calculate the answer
    answer = 1
    for component in components:
        answer = (answer * max_val[component[0]]) % MOD
    
    print(answer)

solve()