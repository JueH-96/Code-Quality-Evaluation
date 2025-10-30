import sys

def main():
    """
    This is the main function that executes the solution logic.
    """
    # Read N, X, and the list of song durations T from standard input.
    try:
        N, X = map(int, sys.stdin.readline().split())
        T = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        # According to problem constraints, input will always be valid.
        return

    # The modulus for all calculations.
    MOD = 998244353

    # We need to compute probabilities, which involves division. In modular arithmetic,
    # division by N is multiplication by the modular multiplicative inverse of N.
    # We can calculate this using Fermat's Little Theorem: N^(MOD-2) ≡ N^(-1) (mod MOD)
    # since MOD is a prime number.
    inv_N = pow(N, MOD - 2, MOD)

    # We use dynamic programming. Let dp[t] be the probability that the total
    # duration of a sequence of randomly chosen songs is exactly t.
    dp = [0] * (X + 1)

    # The base case is at time 0. No songs have been played, so the total duration is 0.
    # This is our starting state, which occurs with a probability of 1.
    dp[0] = 1

    # We fill the dp table from t = 1 up to X.
    # The recurrence relation is:
    # dp[t] = Σ (over all songs i) P(total time was t - T_i) * P(song i is chosen next)
    # dp[t] = Σ (dp[t - T_i] * 1/N)
    # dp[t] = (1/N) * Σ dp[t - T_i]
    for t in range(1, X + 1):
        current_sum = 0
        for duration in T:
            # We can only transition from a state with total time t-duration
            # if t-duration is non-negative.
            if t >= duration:
                current_sum = (current_sum + dp[t - duration]) % MOD
        dp[t] = (current_sum * inv_N) % MOD
    
    # We want to find the probability that song 1 is playing at time X + 0.5.
    # Let the total duration of songs played before the current song be S.
    # The current song (song 1, with duration T_1) will be playing at time X + 0.5 if:
    # S <= X + 0.5 < S + T_1
    # Since S and T_1 are integers, this is equivalent to:
    # X - T_1 < S <= X
    T_1 = T[0]

    # We need to sum the probabilities of all states S that satisfy this condition.
    prob_sum = 0
    # The range for S is from max(0, X - T_1 + 1) to X.
    start_S_range = max(0, X - T_1 + 1)
    for S in range(start_S_range, X + 1):
        prob_sum = (prob_sum + dp[S]) % MOD
        
    # The final probability is the sum of probabilities of reaching a valid state S,
    # multiplied by the probability of choosing song 1 next, which is 1/N.
    ans = (prob_sum * inv_N) % MOD
    
    print(ans)

if __name__ == "__main__":
    main()