import sys

def power(a, b, m):
    """Calculates (a^b) % m using modular exponentiation."""
    res = 1
    a %= m
    while b > 0:
        if b % 2 == 1:
            res = (res * a) % m
        a = (a * a) % m
        b //= 2
    return res

def modInverse(n, m):
    """Calculates modular inverse of n modulo m using Fermat's Little Theorem.
    m must be a prime number and n must not be divisible by m."""
    # Fermat's Little Theorem: if m is prime, a^(m-2) = a^(-1) (mod m) for a != 0 (mod m)
    return power(n, m - 2, m)

def solve():
    # Read N, X
    line1 = sys.stdin.readline().split()
    N = int(line1[0])
    X = int(line1[1])
    
    # Read song durations T_1, ..., T_N
    # Use 0-based indexing for T, so T[0] is duration of song 1
    T = list(map(int, sys.stdin.readline().split()))
    
    MOD = 998244353
    
    # Calculate modular inverse of N
    N_inv = modInverse(N, MOD)
    
    # dp[i] will store the probability that the sum of durations of songs played so far is exactly i
    # We need dp values up to X, as the start time S for song 1 can be at most X.
    dp = [0] * (X + 1)
    
    # Base case: empty sequence has sum 0 with probability 1
    # This represents the state at time 0, before any song is played.
    dp[0] = 1
    
    # Compute DP table up to X
    # dp[i] = Sum_{j=0}^{N-1} P(total duration before song j was i - T[j] AND song j was chosen)
    # P(total duration before song j was i - T[j]) is approximately dp[i - T[j]]
    # P(song j was chosen) = 1/N (independent of previous sequence)
    # dp[i] = Sum_{j=0}^{N-1} dp[i - T[j]] * (1/N) for i > 0
    for i in range(1, X + 1):
        current_dp_sum = 0
        for j in range(N):
            # If we can reach time i by adding song j (duration T[j]) to a state with total duration i - T[j]
            if i - T[j] >= 0:
                current_dp_sum = (current_dp_sum + dp[i - T[j]]) % MOD
        # Multiply the sum by 1/N (the probability of choosing any specific song)
        dp[i] = (current_dp_sum * N_inv) % MOD

    # We want the probability that song 1 is playing at time X + 0.5.
    # This happens if song 1 starts playing at time S and X + 0.5 falls within [S, S + T_1).
    # S is the total duration of songs played *before* the current song (which is song 1).
    # S must be an integer sum of durations of previous songs, so S >= 0.
    # The condition S <= X + 0.5 < S + T_1 for an integer S is equivalent to:
    # 1. S <= X + 0.5  => S <= floor(X + 0.5) = X (since S is integer and X is integer)
    # 2. X + 0.5 < S + T_1 => X + 0.5 - T_1 < S
    # Combining these, we need an integer S such that X + 0.5 - T_1 < S <= X.
    # The smallest such integer S is floor(X + 0.5 - T_1) + 1.
    # Since X and T_1 are integers, X - T_1 is an integer. X + 0.5 - T_1 = (X - T_1) + 0.5.
    # floor((X - T_1) + 0.5) + 1 = (X - T_1) + 1.
    # So the range for S is [X - T_1 + 1, X].
    # Since S must be non-negative, the actual range is [max(0, X - T_1 + 1), X].
    
    T1 = T[0] # Duration of song 1
    S_min = max(0, X - T1 + 1)
    S_max = X
    
    total_sum_dp_S = 0
    
    # Sum dp[S] for S in the valid range [S_min, S_max].
    # dp[S] is the probability that the total duration of songs played *before* the current song is S.
    # If S_min > S_max, the range is empty, sum is 0. This is handled by the range() function.
    for S in range(S_min, S_max + 1):
        total_sum_dp_S = (total_sum_dp_S + dp[S]) % MOD
        
    # The probability that song 1 starts exactly at time S is P(sum of previous durations is S) * P(next song is song 1)
    # This is dp[S] * (1/N).
    # The total probability that song 1 is playing at time X + 0.5 is the sum of these probabilities for valid S.
    # Total probability = Sum_{S=S_min}^{S_max} dp[S] * (1/N)
    # Total probability = (Sum_{S=S_min}^{S_max} dp[S]) * (1/N)
    
    final_probability = (total_sum_dp_S * N_inv) % MOD
    
    print(final_probability)

solve()