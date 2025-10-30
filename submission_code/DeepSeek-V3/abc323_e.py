MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    X = int(data[1])
    T = list(map(int, data[2:2+N]))
    
    # Precompute the total sum of T_i
    total_T = sum(T)
    
    # Compute the probability that song 1 is being played at time X + 0.5
    # The probability is the sum over all possible sequences of songs that end with song 1 and the total time is X + 0.5 - t1/2
    
    # We need to find all possible sequences of songs such that the sum of their durations is X + 0.5 - t1/2
    
    # Since the songs are chosen uniformly at random, the probability is the sum of (1/N)^k for all sequences of length k that satisfy the condition
    
    # To simplify, we can model this as a dynamic programming problem where we compute the probability that the total time is X + 0.5 - t1/2
    
    # However, since X can be up to 10^4 and N up to 10^3, a direct DP approach is feasible
    
    # We will use a DP table where dp[t] represents the probability that the total time is t
    
    # Initialize dp[0] = 1
    # For each song, update the dp table
    
    # Since the time is continuous, we need to discretize the time steps. However, since the songs have integer durations, we can consider the total time as integer steps
    
    # The time X + 0.5 is equivalent to X + 0.5 seconds, but since the songs have integer durations, we can consider the time as X seconds plus 0.5 seconds
    
    # The 0.5 seconds is within the duration of the last song, so we need to find the probability that the last song is song 1 and the total time before the last song is X - t1/2
    
    # Since t1 is the duration of song 1, t1/2 is not necessarily an integer, but since the songs have integer durations, the total time before the last song must be X - t1/2
    
    # To handle this, we can consider the total time as X + 0.5 seconds, and the last song must be song 1, and the total time before the last song is X + 0.5 - t1
    
    # So, the total time before the last song is X + 0.5 - t1
    
    # Since the songs have integer durations, X + 0.5 - t1 must be an integer
    
    # Therefore, t1 must be such that X + 0.5 - t1 is an integer
    
    # This implies that t1 must be odd if X is even, and t1 must be even if X is odd
    
    # So, we need to consider only the songs where t1 satisfies this condition
    
    # Now, we can proceed with the DP approach
    
    # Initialize the DP table
    max_time = X + 1
    dp = [0] * (max_time + 1)
    dp[0] = 1
    
    # Iterate over each song
    for t in T:
        new_dp = [0] * (max_time + 1)
        for i in range(max_time + 1):
            if dp[i]:
                if i + t <= max_time:
                    new_dp[i + t] = (new_dp[i + t] + dp[i] * pow(N, MOD-2, MOD)) % MOD
        dp = new_dp
    
    # Now, we need to find the probability that the last song is song 1 and the total time before the last song is X + 0.5 - t1
    
    # Since t1 is the duration of song 1, and the total time before the last song is X + 0.5 - t1
    
    # We need to find all t1 such that X + 0.5 - t1 is an integer
    
    # So, t1 must be such that X + 0.5 - t1 is an integer
    
    # This implies that t1 must be odd if X is even, and t1 must be even if X is odd
    
    # So, we need to consider only the songs where t1 satisfies this condition
    
    # Now, we can compute the probability
    
    # Initialize the result
    result = 0
    
    # Iterate over all possible t1
    t1 = T[0]
    if (X + 0.5 - t1) >= 0 and (X + 0.5 - t1) == int(X + 0.5 - t1):
        time_before = int(X + 0.5 - t1)
        if time_before <= max_time:
            result = (result + dp[time_before] * pow(N, MOD-2, MOD)) % MOD
    
    # Now, we need to consider all possible sequences where the last song is song 1 and the total time before the last song is X + 0.5 - t1
    
    # Since the DP table already accounts for all possible sequences, we can directly use the DP value
    
    # So, the result is the sum of dp[time_before] * (1/N) for all time_before = X + 0.5 - t1
    
    # Since t1 is fixed as T[0], we only need to compute for t1 = T[0]
    
    # So, the result is dp[time_before] * (1/N)
    
    # Now, we need to compute the modular inverse of N
    
    inv_N = pow(N, MOD-2, MOD)
    
    # Compute the final result
    result = (dp[time_before] * inv_N) % MOD
    
    print(result)

if __name__ == "__main__":
    main()