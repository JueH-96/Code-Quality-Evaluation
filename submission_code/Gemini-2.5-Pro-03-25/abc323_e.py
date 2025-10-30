# YOUR CODE HERE
import sys

# Using fast_pow for modular exponentiation
def fast_pow(base, power):
    """
    Computes (base^power) % MOD efficiently using binary exponentiation.
    """
    MOD = 998244353 # The specified modulus
    result = 1
    base %= MOD # Ensure base is within modulo range initially
    while power > 0:
        # If power is odd, multiply result with base
        if power % 2 == 1:
            result = (result * base) % MOD
        # Square the base and halve the power
        base = (base * base) % MOD
        power //= 2
    return result

# Using modular inverse function based on Fermat's Little Theorem
def inverse(a):
    """
    Computes modular inverse of a modulo MOD using Fermat's Little Theorem.
    Requires MOD to be prime and 'a' not divisible by MOD.
    """
    MOD = 998244353 # The specified modulus
    # Fermat's Little Theorem states that if MOD is prime, then for any integer a not divisible by MOD,
    # a^(MOD-1) % MOD = 1. This implies a * a^(MOD-2) % MOD = 1.
    # Thus, a^(MOD-2) % MOD is the modular inverse of a.
    # The problem constraints guarantee N >= 2, and MOD is prime, so N is not divisible by MOD.
    return fast_pow(a, MOD - 2)

# Define the modulus globally for clarity, although it's hardcoded in functions
MOD = 998244353

# Main logic encapsulated in a function
def solve():
    """
    Reads input, computes the probability using dynamic programming, and prints the result.
    """
    # Read input N (number of songs) and X (time point check)
    N, X = map(int, sys.stdin.readline().split())
    # Read song durations T into a list. T[i] is duration of song i+1 (using 0-based indexing for list).
    T = list(map(int, sys.stdin.readline().split()))

    # Calculate modular inverse of N. This represents 1/N probability factor modulo MOD.
    N_inv = inverse(N)

    # Initialize DP array. dp[t] will store the probability that *any* song starts exactly at time t.
    # The array size needs to be X+1 to cover times from 0 up to X.
    dp = [0] * (X + 1)
    
    # Base case: At time 0, a song must start. The probability of this event is 1.
    dp[0] = 1 

    # Compute DP states iteratively from t=1 up to X.
    for t in range(1, X + 1):
        current_sum = 0 # Accumulator for sum of probabilities of relevant previous states
        
        # Iterate through all N songs to calculate dependencies.
        for j in range(N):
            # A song starting at time t means some song j must have finished at time t.
            # If song j (duration T[j]) finished at time t, it must have started at time t - T[j].
            # We need the probability dp[t - T[j]] that *a* song started at that time.
            
            # Check if the required previous state index t - T[j] is valid (non-negative).
            if t - T[j] >= 0:
                # Add the probability dp[t - T[j]] to the sum.
                # All calculations are done modulo MOD.
                current_sum = (current_sum + dp[t - T[j]]) % MOD
        
        # The recurrence relation for q(t), the probability a song starts at time t, is:
        # q(t) = Sum_{j=1..N} P(song j starts at t-T_j and ends at t)
        # P(song j starts at t-T_j and ends at t) = P(a song starts at t-T_j) * P(song j chosen | a song starts at t-T_j)
        # = q(t-T_j) * (1/N)
        # So, q(t) = Sum_{j=1..N} [ q(t-T_j) * (1/N) ] = (1/N) * Sum_{j=1..N} q(t-T_j)
        # In our DP array terms: dp[t] = N_inv * Sum_{j=0..N-1, if t-T[j]>=0} dp[t-T[j]]
        
        # Apply the recurrence relation modulo MOD.
        dp[t] = (current_sum * N_inv) % MOD

    # --- Final Probability Calculation ---
    # We need the probability that song 1 (index 0, duration T[0]) is playing at time X + 0.5.
    # A song is playing at time X + 0.5 if it started at some integer time t_s
    # such that t_s <= X + 0.5 < t_s + duration.
    # For song 1 with duration T1 = T[0], this means: t_s <= X + 0.5 < t_s + T1.
    # Since t_s must be an integer, this simplifies to: X - T1 + 1 <= t_s <= X.
    
    total_prob_sum_q = 0 # Accumulator for the sum of probabilities q(t_s) over the valid range of t_s.
    
    # Duration of song 1
    T1 = T[0] 
    # Determine the valid range of start times [lower_bound, X] for song 1.
    # Lower bound is max(0, X - T1 + 1) because start time cannot be negative.
    lower_bound = X - T1 + 1
    if lower_bound < 0:
         lower_bound = 0 

    # Sum the probabilities dp[t_s] (which represents q(t_s)) for all valid start times t_s.
    for t_s in range(lower_bound, X + 1):
        # dp[t_s] is the probability that *any* song starts at time t_s.
        total_prob_sum_q = (total_prob_sum_q + dp[t_s]) % MOD

    # The probability that song 1 specifically starts at time t_s is q(t_s) * (1/N).
    # The total probability is the sum of these probabilities over the valid range of t_s:
    # P = Sum_{t_s=lower_bound..X} [q(t_s) * (1/N)]
    # This can be calculated as ( Sum_{t_s=lower_bound..X} q(t_s) ) * (1/N).
    # Perform the final multiplication by N_inv modulo MOD.
    final_prob = (total_prob_sum_q * N_inv) % MOD
    
    # Print the final computed probability modulo MOD.
    print(final_prob)

# Execute the main logic function
solve()