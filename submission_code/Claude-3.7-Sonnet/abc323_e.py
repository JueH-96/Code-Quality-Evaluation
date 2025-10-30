def solve(N, X, T):
    MOD = 998244353
    inv_N = pow(N, MOD - 2, MOD)  # Compute 1/N in modular arithmetic
    
    # dp[t] = probability that a song finishes at time t
    dp = [0] * (X + 1)
    dp[0] = 1  # We start choosing a song at time 0
    
    # Fill in the DP table
    for t in range(X + 1):
        if dp[t] == 0:
            continue
        
        # For each possible song choice
        for i in range(N):
            next_t = t + T[i]
            if next_t <= X:
                dp[next_t] = (dp[next_t] + dp[t] * inv_N) % MOD
    
    # Compute probability song 1 is playing at time (X + 0.5)
    # This happens if song 1 was chosen between max(0, X-T[0]+1) and X inclusive
    probability = 0
    for t in range(max(0, X - T[0] + 1), X + 1):
        probability = (probability + dp[t] * inv_N) % MOD
    
    return probability

# Read input
N, X = map(int, input().split())
T = list(map(int, input().split()))

# Output the result
print(solve(N, X, T))