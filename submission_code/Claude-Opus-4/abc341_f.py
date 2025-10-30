def solve():
    # Read input
    N, M = map(int, input().split())
    
    # Build adjacency list
    adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        u, v = map(int, input().split())
        adj[u].append(v)
        adj[v].append(u)
    
    W = [0] + list(map(int, input().split()))
    A = [0] + list(map(int, input().split()))
    
    # For each vertex, compute the maximum operations we can get
    # when we remove one piece from it
    max_ops = [0] * (N + 1)
    
    # Process vertices in decreasing order of weight
    vertices = list(range(1, N + 1))
    vertices.sort(key=lambda x: -W[x])
    
    for v in vertices:
        # For vertex v, find the best subset of neighbors to distribute to
        neighbors = []
        for u in adj[v]:
            if W[u] < W[v]:
                neighbors.append((W[u], max_ops[u], u))
        
        # Sort neighbors by their potential (max_ops) in descending order
        neighbors.sort(key=lambda x: -x[1])
        
        # Greedily select neighbors while sum of weights < W[v]
        current_sum = 0
        ops_from_v = 1  # 1 for removing the piece from v
        
        for w, ops, u in neighbors:
            if current_sum + w < W[v]:
                current_sum += w
                ops_from_v += ops
        
        max_ops[v] = ops_from_v
    
    # Calculate total operations
    total_ops = 0
    for i in range(1, N + 1):
        total_ops += A[i] * max_ops[i]
    
    print(total_ops)

solve()