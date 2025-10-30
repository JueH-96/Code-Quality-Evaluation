def solve():
    n, m = map(int, input().split())
    
    # Build adjacency list
    adj = [[] for _ in range(n + 1)]
    for _ in range(m):
        u, v = map(int, input().split())
        adj[u].append(v)
        adj[v].append(u)
    
    # Find bipartition using DFS
    color = [-1] * (n + 1)
    
    def dfs(node, c):
        color[node] = c
        for neighbor in adj[node]:
            if color[neighbor] == -1:
                dfs(neighbor, 1 - c)
    
    # Color all connected components
    for i in range(1, n + 1):
        if color[i] == -1:
            dfs(i, 0)
    
    # Count vertices in each partition
    partition_0 = sum(1 for i in range(1, n + 1) if color[i] == 0)
    partition_1 = n - partition_0
    
    # Maximum edges in complete bipartite graph
    max_edges = partition_0 * partition_1
    
    # Number of edges that can be added
    moves_remaining = max_edges - m
    
    # If odd number of moves, first player (Aoki) wins
    if moves_remaining % 2 == 1:
        print("Aoki")
    else:
        print("Takahashi")

solve()