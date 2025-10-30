def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    edges = data[2:2*M+2]
    W = list(map(int, data[2*M+2:2*M+N+2]))
    A = list(map(int, data[2*M+N+2:2*M+2*N+2]))
    
    # Create adjacency list
    adj = [[] for _ in range(N+1)]
    for i in range(M):
        u = int(edges[2*i])
        v = int(edges[2*i+1])
        adj[u].append(v)
        adj[v].append(u)
    
    # Sort vertices by decreasing W_i
    sorted_vertices = sorted(range(1, N+1), key=lambda x: -W[x])
    
    # Initialize dp array
    dp = [0] * (N+1)
    
    for x in sorted_vertices:
        neighbors = [y for y in adj[x] if W[y] < W[x]]
        dp_sum = [0] * W[x]
        for y in neighbors:
            for k in range(W[x]-1, W[y]-1, -1):
                dp_sum[k] = max(dp_sum[k], dp_sum[k - W[y]] + dp[y])
            if W[y] <= W[x] - 1:
                dp_sum[W[y]-1] = max(dp_sum[W[y]-1], dp[y])
        dp[x] = 1 + dp_sum[-1]
    
    # Calculate total operations
    total_operations = sum(A[i] * dp[i] for i in range(1, N+1))
    print(total_operations)

if __name__ == '__main__':
    main()