# YOUR CODE HERE
def solve():
    MOD = 998244353
    
    # Read input
    N, M, K = map(int, input().split())
    
    # Build adjacency list
    adj = [[] for _ in range(N + 1)]
    
    # Add cycle edges: i -> i+1
    for i in range(1, N + 1):
        next_vertex = i + 1 if i < N else 1
        adj[i].append(next_vertex)
    
    # Add additional edges
    for _ in range(M):
        x, y = map(int, input().split())
        adj[x].append(y)
    
    # Dynamic programming
    # dp[v] = number of ways to reach vertex v
    dp = [0] * (N + 1)
    dp[1] = 1  # Start at vertex 1
    
    # For each step
    for step in range(K):
        new_dp = [0] * (N + 1)
        
        # For each vertex
        for u in range(1, N + 1):
            if dp[u] > 0:
                # For each outgoing edge from u
                for v in adj[u]:
                    new_dp[v] = (new_dp[v] + dp[u]) % MOD
        
        dp = new_dp
    
    # Sum up all ways to be at any vertex after K steps
    result = sum(dp[1:]) % MOD
    print(result)

solve()