# YOUR CODE HERE
def solve():
    # Read input
    N, M = map(int, input().split())
    wheels = []
    
    for _ in range(N):
        line = list(map(int, input().split()))
        C = line[0]
        P = line[1]
        S = line[2:2+P]
        wheels.append((C, P, S))
    
    # Dynamic programming
    # dp[m] = minimum expected cost to earn at least m more points
    dp = [float('inf')] * (M + 1)
    dp[0] = 0
    
    # Iterate until convergence
    for iteration in range(1000):  # Sufficient iterations for convergence
        new_dp = dp.copy()
        
        for m in range(1, M + 1):
            min_cost = float('inf')
            
            # Try each wheel
            for i in range(N):
                C, P, S = wheels[i]
                
                # Calculate expected cost for this wheel
                expected_cost = C
                for j in range(P):
                    points_earned = S[j]
                    remaining_points = max(0, m - points_earned)
                    expected_cost += dp[remaining_points] / P
                
                min_cost = min(min_cost, expected_cost)
            
            new_dp[m] = min_cost
        
        # Check convergence
        converged = True
        for m in range(M + 1):
            if abs(new_dp[m] - dp[m]) > 1e-9:
                converged = False
                break
        
        dp = new_dp
        
        if converged:
            break
    
    print(dp[M])

solve()