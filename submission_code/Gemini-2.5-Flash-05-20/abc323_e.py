import sys

def solve():
    # Read N and X
    N, X = map(int, sys.stdin.readline().split())
    # Read the durations of the N songs
    T = list(map(int, sys.stdin.readline().split()))

    MOD = 998244353

    # T_0 is the duration of Song 1 (assuming 0-indexed T corresponds to 1-indexed songs)
    T_0 = T[0]

    # Calculate modular inverse of N using Fermat's Little Theorem: N^(MOD-2) % MOD
    N_inv = pow(N, MOD - 2, MOD)

    # Determine the maximum duration among all songs
    # This is used to set the upper bound for the DP array size.
    max_T = 0
    for t_val in T:
        if t_val > max_T:
            max_T = t_val
    
    # max_dp_idx is the maximum time value we need to track in our dp array.
    # A song can start at time 'k' (up to X) and have duration max_T.
    # So, the cumulative time can go up to X + max_T.
    max_dp_idx = X + max_T 

    # dp[k] will store the probability that the sum of durations of songs played so far is exactly k.
    # The array needs to be of size max_dp_idx + 1 to include index max_dp_idx.
    dp = [0] * (max_dp_idx + 1)

    # Base case: At time 0, no songs have been played, so the probability of being at time 0 is 1.
    dp[0] = 1

    # Fill the dp table using dynamic programming.
    # Iterate through each possible cumulative time 'k' that has a non-zero probability.
    # 'k' goes from 0 up to max_dp_idx - 1.
    # This ensures that when we process dp[k], all required dp[k'] for k' < k are already correctly computed.
    for k in range(max_dp_idx):
        if dp[k] == 0:
            continue # If there's no way to reach time k, no new paths can start from here.
        
        # From time 'k', choose any song with duration 't_j'.
        # This transitions to a new cumulative time 'k + t_j'.
        for t_j in T:
            next_time = k + t_j
            # Only update if the next_time is within our tracking range.
            if next_time <= max_dp_idx:
                dp[next_time] = (dp[next_time] + dp[k] * N_inv) % MOD

    # Calculate the total probability that Song 1 is being played at X + 0.5 seconds.
    total_prob = 0
    # Iterate through all possible start times 'k' for Song 1.
    # Song 1 must start at an integer time 'k' such that 0 <= k <= X.
    # If it starts after X, it cannot be playing at X + 0.5.
    for k in range(X + 1): # k iterates from 0 to X
        # For Song 1 (duration T_0) to be playing at X + 0.5 seconds,
        # it must have started at 'k' and its duration must extend beyond X.
        # This means k + T_0 > X.
        if k + T_0 > X:
            # The probability of reaching time 'k' is dp[k].
            # The probability of choosing Song 1 as the next song is 1/N.
            # Multiply these probabilities and add to the total.
            total_prob = (total_prob + dp[k] * N_inv) % MOD
            
    # Print the final probability modulo 998244353.
    sys.stdout.write(str(total_prob) + "
")

# Call the solve function to execute the program.
solve()