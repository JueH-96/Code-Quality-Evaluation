def solve():
    MOD = 998244353
    
    N, M, K = map(int, input().split())
    
    # Build adjacency list representation
    adj = [[] for _ in range(N+1)]
    
    # Add cycle edges
    for i in range(1, N):
        adj[i].append(i+1)
    adj[N].append(1)
    
    # Add additional edges
    for _ in range(M):
        x, y = map(int, input().split())
        adj[x].append(y)
        
    # dp[k][v] = number of ways to reach vertex v after k moves
    dp = [[0]*(N+1) for _ in range(K+1)]
    dp[0][1] = 1
    
    # For each number of moves
    for k in range(K):
        # For each current vertex
        for v in range(1, N+1):
            # For each neighbor of current vertex
            for nxt in adj[v]:
                dp[k+1][nxt] = (dp[k+1][nxt] + dp[k][v]) % MOD
                
    print(dp[K][1])

solve()