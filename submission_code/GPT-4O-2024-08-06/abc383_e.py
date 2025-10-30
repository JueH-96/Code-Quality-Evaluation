def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    N = int(data[index])
    index += 1
    M = int(data[index])
    index += 1
    K = int(data[index])
    index += 1
    
    edges = []
    for _ in range(M):
        u = int(data[index]) - 1
        index += 1
        v = int(data[index]) - 1
        index += 1
        w = int(data[index])
        index += 1
        edges.append((w, u, v))
    
    A = []
    for _ in range(K):
        A.append(int(data[index]) - 1)
        index += 1
    
    B = []
    for _ in range(K):
        B.append(int(data[index]) - 1)
        index += 1
    
    # Sort edges by weight
    edges.sort()
    
    # Union-Find data structure
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
    
    # Minimum path weight between any two nodes
    min_path_weight = [[float('inf')] * N for _ in range(N)]
    
    for w, u, v in edges:
        if find(u) != find(v):
            union(u, v)
            for i in range(N):
                for j in range(N):
                    if find(i) == find(j):
                        min_path_weight[i][j] = min(min_path_weight[i][j], w)
    
    # Calculate the minimum sum of f(A_i, B_i) by sorting A and B
    A.sort()
    B.sort()
    
    result = 0
    for i in range(K):
        result += min_path_weight[A[i]][B[i]]
    
    print(result)