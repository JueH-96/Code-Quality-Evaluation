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
    
    # Compute the probability that song 1 is being played at (X + 0.5) seconds
    # The probability is the sum over all possible sequences of songs that end with song 1 and the total time is X + 0.5 seconds
    
    # We need to find all possible sequences of songs such that the sum of their durations is X + 0.5 seconds, and the last song is song 1
    
    # Since the songs are chosen uniformly at random, the probability is the sum of (1/N)^k for all sequences of length k that satisfy the condition
    
    # To compute this, we can use dynamic programming
    
    # Initialize dp[t] as the probability that the total time is t seconds and the last song is song 1
    # We need to find dp[X + 0.5]
    
    # Since X is an integer and T_i are integers, X + 0.5 is not an integer, so we need to consider the fractional part
    
    # However, since the songs are played continuously, the time X + 0.5 seconds corresponds to the time when a song is being played, not when it starts or ends
    
    # So, we need to find the probability that at time X + 0.5 seconds, song 1 is being played
    
    # This can be rephrased as the probability that the total time up to the start of the last song is <= X + 0.5 < total time up to the end of the last song, and the last song is song 1
    
    # So, we need to find the probability that the total time up to the start of the last song is <= X + 0.5 < total time up to the end of the last song, and the last song is song 1
    
    # Let's define dp[t] as the probability that the total time is t seconds and the last song is song 1
    
    # Then, the probability that song 1 is being played at X + 0.5 seconds is the sum of dp[t] for all t such that t <= X + 0.5 < t + T_1
    
    # Since X + 0.5 is not an integer, we need to consider the fractional part
    
    # However, since T_i are integers, the only way t <= X + 0.5 < t + T_1 is if t <= X < t + T_1
    
    # So, we need to find the sum of dp[t] for all t such that t <= X < t + T_1
    
    # Now, we need to compute dp[t] for all t up to X + T_1
    
    # Initialize dp[0] = 0, since no songs have been played yet
    
    # For each t, and for each song i, we update dp[t + T_i] += dp[t] / N
    
    # Additionally, if i == 1, we need to consider that the last song is song 1
    
    # So, we need to keep track of the last song
    
    # Let's define dp[t][i] as the probability that the total time is t seconds and the last song is song i
    
    # Initialize dp[0][i] = 0 for all i
    
    # For each t, and for each song j, we update dp[t + T_j][j] += dp[t][i] / N for all i
    
    # Then, the probability that song 1 is being played at X + 0.5 seconds is the sum of dp[t][1] for all t such that t <= X < t + T_1
    
    # Initialize dp[0][i] = 0 for all i
    
    # Since the songs are chosen uniformly at random, the initial state is that no songs have been played yet
    
    # So, the initial state is t = 0, and no song has been played
    
    # We can represent this as dp[0][i] = 0 for all i
    
    # Now, we need to compute dp[t][i] for all t up to X + T_1
    
    # Initialize dp[0][i] = 0 for all i
    
    # For each t from 0 to X + T_1 - 1, and for each song j, we update dp[t + T_j][j] += dp[t][i] / N for all i
    
    # Additionally, we need to consider the initial state where the first song is chosen
    
    # So, for each song j, we initialize dp[T_j][j] = 1 / N
    
    # Now, we can proceed with the dynamic programming
    
    # Initialize dp[t][i] as a dictionary or list of lists
    
    # Since t can be up to X + T_1, and X can be up to 10^4, and T_1 can be up to 10^4, t can be up to 2 * 10^4
    
    # So, we can use a list of lists with size (X + T_1 + 1) x N
    
    # Initialize dp[t][i] = 0 for all t and i
    
    dp = [[0] * N for _ in range(X + T[0] + 1)]
    
    # Initialize the first step
    for j in range(N):
        if T[j] <= X + T[0]:
            dp[T[j]][j] = 1 / N
    
    # Now, perform the dynamic programming
    for t in range(X + T[0]):
        for i in range(N):
            if dp[t][i] == 0:
                continue
            for j in range(N):
                if t + T[j] <= X + T[0]:
                    dp[t + T[j]][j] = (dp[t + T[j]][j] + dp[t][i] / N) % MOD
    
    # Now, compute the probability that song 1 is being played at X + 0.5 seconds
    # This is the sum of dp[t][1] for all t such that t <= X < t + T[0]
    
    prob = 0
    for t in range(X + 1):
        if t <= X < t + T[0]:
            prob = (prob + dp[t][0]) % MOD
    
    # Since the probability is y / x, and x is not divisible by MOD, we need to find z such that x * z ≡ y mod MOD
    
    # Here, prob is y / x, but since we have been working with fractions, we need to represent it as a fraction
    
    # However, in the dynamic programming, we have been multiplying by 1 / N, which is a fraction
    
    # To handle this, we can represent the probabilities as fractions with denominator N^k, where k is the number of steps
    
    # But since the number of steps can be up to X + T[0], which is up to 2 * 10^4, it's not feasible to keep track of the denominator
    
    # Instead, we can represent the probabilities as integers modulo MOD, and use the fact that 1 / N ≡ N^(MOD-2) mod MOD
    
    # So, in the dynamic programming, we can multiply by inv_N, where inv_N is the modular inverse of N
    
    # Let's recompute the dynamic programming with this approach
    
    inv_N = pow(N, MOD - 2, MOD)
    
    dp = [[0] * N for _ in range(X + T[0] + 1)]
    
    for j in range(N):
        if T[j] <= X + T[0]:
            dp[T[j]][j] = inv_N
    
    for t in range(X + T[0]):
        for i in range(N):
            if dp[t][i] == 0:
                continue
            for j in range(N):
                if t + T[j] <= X + T[0]:
                    dp[t + T[j]][j] = (dp[t + T[j]][j] + dp[t][i] * inv_N) % MOD
    
    prob = 0
    for t in range(X + 1):
        if t <= X < t + T[0]:
            prob = (prob + dp[t][0]) % MOD
    
    # Now, prob is y / x, but since we have been working modulo MOD, prob is y * inv_x mod MOD
    
    # To find z such that x * z ≡ y mod MOD, we need to find z ≡ y * inv_x mod MOD
    
    # But since x is not divisible by MOD, and prob is y * inv_x mod MOD, we can directly output prob
    
    print(prob)

if __name__ == "__main__":
    main()