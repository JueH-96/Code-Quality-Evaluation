def solve():
    # Read input
    N, M = map(int, input().split())
    edges = []
    for _ in range(M):
        u, v = map(int, input().split())
        edges.append((u-1, v-1))  # Convert to 0-indexed
    
    W = list(map(int, input().split()))
    A = list(map(int, input().split()))
    
    # Build adjacency list
    adj = [[] for _ in range(N)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Initialize p values
    p = [1] * N  # Initial guess: each piece contributes 1 operation
    
    # Iteratively update p values until convergence
    for iteration in range(100):  # Limit iterations
        new_p = [0] * N
        
        for x in range(N):
            max_ops = 1  # At least 1 operation (remove the piece)
            
            neighbors = adj[x]
            if neighbors:
                # Use DP to find maximum value for valid subsets
                target = W[x] - 1  # We want sum < W[x], so sum <= W[x] - 1
                dp = {0: 0}  # dp[sum] = maximum value achievable with sum
                
                for y in neighbors:
                    new_dp = {}
                    for s, v in dp.items():
                        # Option 1: don't include y
                        if s not in new_dp or new_dp[s] < v:
                            new_dp[s] = v
                        # Option 2: include y (if weight constraint satisfied)
                        if s + W[y] <= target:
                            new_sum = s + W[y]
                            new_value = v + p[y]
                            if new_sum not in new_dp or new_dp[new_sum] < new_value:
                                new_dp[new_sum] = new_value
                    dp = new_dp
                
                # Find the maximum value among all valid subsets
                max_value = max(dp.values()) if dp else 0
                max_ops = 1 + max_value
            
            new_p[x] = max_ops
        
        # Check for convergence
        if new_p == p:
            break
        p = new_p
    
    # Compute the answer
    result = sum(A[i] * p[i] for i in range(N))
    print(result)

solve()