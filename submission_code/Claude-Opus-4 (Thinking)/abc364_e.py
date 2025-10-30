N, X, Y = map(int, input().split())
dishes = []
for _ in range(N):
    a, b = map(int, input().split())
    dishes.append((a, b))

# dp[s][t] = maximum number of dishes to reach sweetness s and saltiness t
# Initialize with -1 (unreachable)
dp = [[-1 for _ in range(Y+1)] for _ in range(X+1)]
dp[0][0] = 0

# Process each dish
for a, b in dishes:
    # Create a copy to avoid using same dish multiple times
    new_dp = [row[:] for row in dp]
    
    for s in range(X+1):
        for t in range(Y+1):
            if dp[s][t] != -1:  # If this state is reachable
                # Try adding current dish
                if s + a <= X and t + b <= Y:
                    new_dp[s+a][t+b] = max(new_dp[s+a][t+b], dp[s][t] + 1)
    
    dp = new_dp

# Find maximum dishes eaten within limits
ans = 0
for s in range(X+1):
    for t in range(Y+1):
        if dp[s][t] != -1:
            ans = max(ans, dp[s][t])

# Check if we can eat one more dish that exceeds limits
# For each reachable state, try adding any dish that would exceed limits
for s in range(X+1):
    for t in range(Y+1):
        if dp[s][t] != -1 and dp[s][t] > 0:  # Valid state with at least 1 dish
            # Try each dish to see if it would exceed limits
            for a, b in dishes:
                if s + a > X or t + b > Y:
                    # This dish would exceed limits, so we can eat dp[s][t] + 1 dishes total
                    ans = max(ans, dp[s][t] + 1)

print(ans)