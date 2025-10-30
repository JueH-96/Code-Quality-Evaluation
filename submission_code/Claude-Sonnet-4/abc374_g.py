def solve():
    n = int(input())
    products = []
    for _ in range(n):
        products.append(input().strip())
    
    # Build directed graph
    adj = [[] for _ in range(n)]
    in_degree = [0] * n
    
    for i in range(n):
        for j in range(n):
            if i != j and products[i][1] == products[j][0]:
                adj[i].append(j)
                in_degree[j] += 1
    
    # Count nodes with in_degree 0
    sources = sum(1 for i in range(n) if in_degree[i] == 0)
    
    # If there are no sources, we have only cycles
    if sources == 0:
        # Find number of strongly connected components
        visited = [False] * n
        components = 0
        
        def dfs(node):
            visited[node] = True
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    dfs(neighbor)
        
        for i in range(n):
            if not visited[i]:
                dfs(i)
                components += 1
        
        print(components)
    else:
        print(sources)

solve()