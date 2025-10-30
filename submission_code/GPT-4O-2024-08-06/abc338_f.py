def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    edges = []
    index = 2
    for _ in range(M):
        U = int(data[index]) - 1
        V = int(data[index + 1]) - 1
        W = int(data[index + 2])
        edges.append((U, V, W))
        index += 3
    
    # Initialize the DP table
    INF = float('inf')
    dp = [[INF] * N for _ in range(1 << N)]
    
    # Base case: start from each vertex
    for i in range(N):
        dp[1 << i][i] = 0
    
    # Fill the DP table
    for mask in range(1 << N):
        for u in range(N):
            if dp[mask][u] == INF:
                continue
            for (U, V, W) in edges:
                if U == u and not (mask & (1 << V)):
                    new_mask = mask | (1 << V)
                    dp[new_mask][V] = min(dp[new_mask][V], dp[mask][u] + W)
    
    # Find the minimum weight walk that visits all vertices
    full_mask = (1 << N) - 1
    result = min(dp[full_mask][v] for v in range(N))
    
    if result == INF:
        print("No")
    else:
        print(result)