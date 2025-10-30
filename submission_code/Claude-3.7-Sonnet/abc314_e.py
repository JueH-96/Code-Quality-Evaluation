def solve(N, M, wheels):
    # Initialize DP array
    dp = [0] * (M + 1)  # dp[p] = expected cost to reach M points starting with p points
    
    # Value iteration
    epsilon = 1e-10  # Convergence threshold
    iterations = 0
    max_iterations = 1000000  # Maximum iterations to avoid infinite loops
    
    while iterations < max_iterations:
        old_dp = dp.copy()
        
        # Update expected costs
        for p in range(M):
            min_expected_cost = float('inf')
            for i in range(N):
                C_i, P_i, S_i = wheels[i]
                expected_cost = C_i
                for j in range(P_i):
                    next_p = min(p + S_i[j], M)
                    expected_cost += (1 / P_i) * old_dp[next_p]
                min_expected_cost = min(min_expected_cost, expected_cost)
            
            dp[p] = min_expected_cost
        
        # Check convergence
        max_diff = 0
        for p in range(M):
            max_diff = max(max_diff, abs(dp[p] - old_dp[p]))
        
        if max_diff < epsilon:
            break
        
        iterations += 1
    
    return dp[0]  # Return the expected cost starting from 0 points

N, M = map(int, input().split())
wheels = []
for _ in range(N):
    parts = list(map(int, input().split()))
    C_i = parts[0]
    P_i = parts[1]
    S_i = parts[2:2+P_i]
    wheels.append((C_i, P_i, S_i))

print(solve(N, M, wheels))