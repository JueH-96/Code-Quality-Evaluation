import sys

def solve():
    N, X = map(int, sys.stdin.readline().split())
    T_lengths = list(map(int, sys.stdin.readline().split()))

    MOD = 998244353

    # Modular inverse of N. Requires Python 3.8+ for pow(N, -1, MOD).
    # Otherwise, use pow(N, MOD - 2, MOD) since MOD is prime.
    N_inv = pow(N, -1, MOD)

    # dp[t]: probability that a sequence of songs ends exactly at time t.
    # A new song is chosen at time t.
    # Base case: dp[0] = 1 (at time 0, before any song, prob of being at this boundary is 1).
    dp = [0] * (X + 1)
    dp[0] = 1

    for t in range(1, X + 1):
        current_sum_prev_dp = 0
        for song_duration in T_lengths:
            if t - song_duration >= 0:
                current_sum_prev_dp = (current_sum_prev_dp + dp[t - song_duration]) % MOD
        
        dp[t] = (current_sum_prev_dp * N_inv) % MOD

    # Song 1 (duration T_lengths[0]) is playing at time X + 0.5 if it started at S such that:
    # max(0, X - T_lengths[0] + 1) <= S <= X
    
    sum_of_relevant_dp_S = 0
    
    T1_duration = T_lengths[0] # Duration of song 1
    
    min_S = max(0, X - T1_duration + 1)
    max_S = X 
    
    # Sum dp[s_start_time] for s_start_time in the valid range [min_S, max_S]
    if min_S <= max_S: # Ensure range is valid before iterating
        for s_start_time in range(min_S, max_S + 1):
            sum_of_relevant_dp_S = (sum_of_relevant_dp_S + dp[s_start_time]) % MOD
            
    # Probability that song 1 starts at S is dp[S] * N_inv.
    # Total probability is (sum of dp[S] for valid S) * N_inv.
    final_ans = (sum_of_relevant_dp_S * N_inv) % MOD
    
    print(final_ans)

solve()