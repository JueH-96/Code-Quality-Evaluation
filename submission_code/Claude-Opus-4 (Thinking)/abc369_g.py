def solve():
    n = int(input())
    graph = [[] for _ in range(n + 1)]
    
    for _ in range(n - 1):
        u, v, l = map(int, input().split())
        graph[u].append((v, l))
        graph[v].append((u, l))
    
    # Root the tree at vertex 1 and collect edge information
    edges = []  # (weight, subtree_vertex_count)
    
    def dfs(u, parent):
        subtree_vertex_count = 1
        for v, weight in graph[u]:
            if v != parent:
                child_subtree_count = dfs(v, u)
                edges.append((weight, child_subtree_count))
                subtree_vertex_count += child_subtree_count
        return subtree_vertex_count
    
    dfs(1, -1)
    
    # DP: dp[k] = maximum total edge weight when selecting exactly k vertices
    # Since vertex 1 is always included but doesn't add edges, we count vertices in subtrees
    dp = [0] * n
    
    # Process each edge - it's like a knapsack problem
    # Each edge can contribute its weight if we select at least 1 vertex from its subtree
    for weight, subtree_count in edges:
        # Process in reverse order to avoid using updated values in same iteration
        for k in range(n - 1, -1, -1):
            # If we include this edge, we must select 1 to subtree_count vertices from its subtree
            for select in range(1, min(subtree_count + 1, n - k)):
                dp[k + select] = max(dp[k + select], dp[k] + weight)
    
    # Output the answer for each K
    for K in range(1, n + 1):
        # We can select at most n-1 vertices from subtrees (all except vertex 1)
        # Vertex 1 is always implicitly included in the minimal subtree
        if K <= n - 1:
            print(2 * dp[K])
        else:
            # If K = n, we must select all vertices
            print(2 * dp[n - 1])

solve()