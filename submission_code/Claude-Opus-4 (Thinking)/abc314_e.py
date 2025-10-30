# YOUR CODE HERE
N, M = map(int, input().split())
wheels = []
for _ in range(N):
    line = list(map(int, input().split()))
    C = line[0]
    P = line[1]
    S = line[2:2+P]
    wheels.append((C, P, S))

# Dynamic programming with value iteration
dp = [0.0] * (M + 1)

# Run value iteration until convergence
for _ in range(10000):
    new_dp = [0.0] * (M + 1)
    
    for m in range(M):
        best_cost = float('inf')
        for C, P, S in wheels:
            expected_cost = C
            for s in S:
                expected_cost += dp[min(M, m + s)] / P
            best_cost = min(best_cost, expected_cost)
        new_dp[m] = best_cost
    
    dp = new_dp

print(dp[0])