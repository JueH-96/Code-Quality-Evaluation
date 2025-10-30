def solve():
    n, q = map(int, input().split())
    
    operations = []
    for i in range(q):
        l, r, c = map(int, input().split())
        operations.append((l, r, c, n + i + 1))  # store hub vertex id
    
    # Check connectivity: union of all ranges must cover [1, N]
    covered = [False] * (n + 1)
    for l, r, c, hub in operations:
        for j in range(l, r + 1):
            covered[j] = True
    
    for j in range(1, n + 1):
        if not covered[j]:
            print(-1)
            return
    
    # Create all edges
    edges = []
    for l, r, c, hub in operations:
        for j in range(l, r + 1):
            edges.append((c, hub, j))
    
    # Sort edges by cost for Kruskal's algorithm
    edges.sort()
    
    # Union-Find for Kruskal's algorithm
    parent = list(range(n + q + 1))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
            return True
        return False
    
    # Kruskal's algorithm
    mst_cost = 0
    edges_added = 0
    target_edges = n + q - 1  # MST has (V-1) edges for V vertices
    
    for cost, u, v in edges:
        if union(u, v):
            mst_cost += cost
            edges_added += 1
            if edges_added == target_edges:
                break
    
    print(mst_cost)

solve()