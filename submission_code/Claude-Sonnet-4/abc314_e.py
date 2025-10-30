import sys

def solve():
    N, M = map(int, input().split())
    
    wheels = []
    for _ in range(N):
        line = list(map(int, input().split()))
        C = line[0]
        P = line[1]
        S = line[2:2+P]
        wheels.append((C, P, S))
    
    # dp[i] = expected cost to reach at least M points when currently having i points
    dp = [0.0] * (M + 1)
    
    # Iterate backwards from M-1 to 0
    for i in range(M-1, -1, -1):
        min_cost = float('inf')
        
        # Try each wheel
        for wheel_idx in range(N):
            C, P, S = wheels[wheel_idx]
            
            # Calculate expected cost for this wheel
            expected_cost = C  # Cost to play this wheel
            
            # Add expected future cost
            for outcome_idx in range(P):
                points_gained = S[outcome_idx]
                new_points = min(i + points_gained, M)
                probability = 1.0 / P
                expected_cost += probability * dp[new_points]
            
            min_cost = min(min_cost, expected_cost)
        
        dp[i] = min_cost
    
    print(dp[0])

solve()