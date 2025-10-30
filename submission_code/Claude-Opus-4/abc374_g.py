# YOUR CODE HERE
def solve():
    n = int(input())
    edges = []
    for _ in range(n):
        s = input().strip()
        edges.append((s[0], s[1]))
    
    # Build adjacency list and calculate degrees
    from collections import defaultdict
    adj = defaultdict(list)
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)
    vertices = set()
    
    for u, v in edges:
        adj[u].append(v)
        out_degree[u] += 1
        in_degree[v] += 1
        vertices.add(u)
        vertices.add(v)
    
    # Count vertices with excess out-degree
    excess_out = 0
    for v in vertices:
        diff = out_degree[v] - in_degree[v]
        if diff > 0:
            excess_out += diff
    
    # If all vertices are balanced, we need to count connected components
    if excess_out == 0:
        # Find connected components in underlying undirected graph
        visited = set()
        components = 0
        
        def dfs(v):
            visited.add(v)
            for u in adj[v]:
                if u not in visited:
                    dfs(u)
            # Also check reverse edges for connectivity
            for u in vertices:
                if v in adj[u] and u not in visited:
                    dfs(u)
        
        for v in vertices:
            if v not in visited:
                components += 1
                dfs(v)
        
        print(components)
    else:
        print(excess_out)

solve()