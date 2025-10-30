import sys
import heapq

input = sys.stdin.read
def solve():
    data = input().split()
    idx = 0
    
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    K = int(data[idx])
    idx += 1
    
    edges = []
    for _ in range(M):
        u = int(data[idx]) - 1
        idx += 1
        v = int(data[idx]) - 1
        idx += 1
        w = int(data[idx])
        idx += 1
        edges.append((w, u, v))
    
    A = []
    for _ in range(K):
        A.append(int(data[idx]) - 1)
        idx += 1
    
    B = []
    for _ in range(K):
        B.append(int(data[idx]) - 1)
        idx += 1
    
    # Sort edges by weight
    edges.sort()
    
    # Kruskal's algorithm to find the Maximum Spanning Forest (MSF) using Union-Find
    parent = list(range(N))
    rank = [0] * N
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        rootX = find(x)
        rootY = find(y)
        if rootX != rootY:
            if rank[rootX] > rank[rootY]:
                parent[rootY] = rootX
            elif rank[rootX] < rank[rootY]:
                parent[rootX] = rootY
            else:
                parent[rootY] = rootX
                rank[rootX] += 1
    
    # Create the Maximum Spanning Forest using Kruskal's algorithm
    max_edge_in_path = {}
    
    for w, u, v in edges:
        if find(u) != find(v):
            union(u, v)
            # Record the maximum edge weight in the path between these two nodes
            max_edge_in_path[(u, v)] = w
            max_edge_in_path[(v, u)] = w
    
    # Use a Union-Find structure to answer the queries
    def find_with_max_edge(x):
        if parent[x] != x:
            original_parent = parent[x]
            parent[x], max_edge = find_with_max_edge(parent[x])
            max_edge_in_path[(x, parent[x])] = max(max_edge_in_path.get((x, original_parent), 0), max_edge)
            max_edge_in_path[(parent[x], x)] = max_edge_in_path[(x, parent[x])]
        return parent[x], max_edge_in_path.get((x, parent[x]), 0)
    
    # Reinitialize the Union-Find for path compression with max edge tracking
    parent = list(range(N))
    for w, u, v in edges:
        if find(u) != find(v):
            union(u, v)
    
    # Calculate f(A_i, B_j) for all i, j
    f = [[0] * K for _ in range(K)]
    for i in range(K):
        for j in range(K):
            if A[i] != B[j]:
                # Find the max edge weight in the path from A[i] to B[j]
                rootA, _ = find_with_max_edge(A[i])
                rootB, _ = find_with_max_edge(B[j])
                if rootA == rootB:
                    # Same component, calculate the path
                    path = []
                    # Move A[i] up to the root
                    x = A[i]
                    while x != rootA:
                        path.append((max_edge_in_path[(x, parent[x])], x, parent[x]))
                        x = parent[x]
                    # Move B[j] up to the root
                    x = B[j]
                    while x != rootB:
                        path.append((max_edge_in_path[(x, parent[x])], x, parent[x]))
                        x = parent[x]
                    # Maximum edge in the path
                    max_edge = max(w for w, _, _ in path)
                    f[i][j] = max_edge
                else:
                    # Different components, should not happen as the graph is connected
                    f[i][j] = float('inf')
    
    # Hungarian algorithm or Min-Cost Max-Flow could be used to solve the assignment problem
    # For simplicity, we use a simpler approach here
    from scipy.optimize import linear_sum_assignment
    cost_matrix = f
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    result = sum(cost_matrix[row_ind[k]][col_ind[k]] for k in range(K))
    
    print(result)