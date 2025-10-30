# YOUR CODE HERE
def solve():
    N, Q = map(int, input().split())
    
    # Read queries
    queries = []
    for _ in range(Q):
        L, R, C = map(int, input().split())
        queries.append((L, R, C))
    
    # Build edge list
    edges = []
    for i in range(Q):
        L, R, C = queries[i]
        for j in range(L, R + 1):
            edges.append((C, j, N + i + 1))
    
    # Union-Find data structure
    parent = list(range(N + Q + 1))
    
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
    
    # Check connectivity by connecting all edges
    for _, u, v in edges:
        union(u, v)
    
    # Check if all vertices are in the same component
    components = set()
    for i in range(1, N + Q + 1):
        components.add(find(i))
    
    if len(components) > 1:
        print(-1)
        return
    
    # Find MST using Kruskal's algorithm
    edges.sort()  # Sort by cost
    parent = list(range(N + Q + 1))  # Reset Union-Find
    mst_cost = 0
    edges_used = 0
    
    for cost, u, v in edges:
        if union(u, v):
            mst_cost += cost
            edges_used += 1
            if edges_used == N + Q - 1:
                break
    
    print(mst_cost)

solve()