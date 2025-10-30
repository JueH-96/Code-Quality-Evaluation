# YOUR CODE HERE
import sys
import math

# Increase recursion depth for safety, although this DP is iterative
# sys.setrecursionlimit(20000) # Not needed for iterative solution

def solve():
    """
    Solves the problem using dynamic programming.
    Reads input N and performances P_1 ... P_N.
    Calculates the maximum possible rating Takahashi can achieve.
    Prints the maximum rating to standard output.
    """
    
    N = int(sys.stdin.readline())
    P = list(map(int, sys.stdin.readline().split()))

    # Precompute powers of 0.9
    # pow09[k] stores (0.9)^k
    # Need up to N-1 power for weights, so size N is sufficient technically. 
    # Size N+1 is safe and convenient for 1-based indexing k.
    pow09 = [1.0] * (N + 1)
    for i in range(1, N + 1):
        pow09[i] = pow09[i-1] * 0.9
    
    # Precompute sums of weights W_k
    # W[k] stores the denominator part of the rating formula for k contests:
    # sum_{j=1}^{k} (0.9)^{k-j} = sum_{m=0}^{k-1} (0.9)^m
    W = [0.0] * (N + 1)
    current_sum_weights = 0.0
    for k in range(1, N + 1):
        # The sum is a geometric series: (0.9)^0 + (0.9)^1 + ... + (0.9)^(k-1)
        current_sum_weights += pow09[k-1] 
        W[k] = current_sum_weights

    # Initialize DP table
    # dp[i][k] stores the maximum numerator value (sum_{j=1}^k (0.9)^{k-j} Q_j)
    # for a sequence of exactly k contests, where the k-th chosen contest 
    # is the i-th overall contest (which has performance P[i-1]).
    # The DP state dimensions are (N+1) x (N+1) to use 1-based indexing for i and k.
    # Initialize with negative infinity to represent invalid or unreachable states.
    # Using a very large negative number like -1e18 might be safer than float('-inf') with arithmetic operations.
    # Let's stick with float('-inf') for conceptual clarity, and use checks like > -1e18 for safety.
    dp = [[-float('inf')] * (N + 1) for _ in range(N + 1)]

    # Initialize overall maximum rating found so far to negative infinity.
    # This ensures that any valid rating found will become the maximum initially.
    max_R = -float('inf') 

    # Iterate over possible sequence lengths k, from 1 to N.
    for k in range(1, N + 1):
        # max_prev_val tracks the maximum value of dp[p][k-1] for contests p < i.
        # This value is crucial for the DP transition.
        # It is reset for each k and updated incrementally within the inner loop over i.
        max_prev_val = -float('inf') 
        
        # Iterate over possible ending contests i (from 1 to N).
        # The i-th contest in the original list has index i-1 and performance P[i-1].
        for i in range(1, N + 1):
            
            # Base case: k = 1
            # A sequence of length 1 ending at contest i.
            # The numerator sum is simply the performance P[i-1].
            if k == 1:
                dp[i][1] = float(P[i-1]) # Convert performance to float for calculations
            else:
                # General case for k > 1.
                # We need the maximum numerator sum from sequences of length k-1 that end at some contest p < i.
                # The variable max_prev_val maintains this maximum value.
                # Update max_prev_val using the result from contest i-1 *before* computing for contest i.
                if i > 1:
                     # max_prev_val currently holds max(dp[p][k-1] for p=1..i-2).
                     # Update it to max(dp[p][k-1] for p=1..i-1) by considering dp[i-1][k-1].
                     max_prev_val = max(max_prev_val, dp[i-1][k-1])

                # Compute dp[i][k] using the recurrence relation derived from the formula:
                # dp[i][k] = P[i-1] + 0.9 * max_{p < i} {dp[p][k-1]}
                # This computation is only possible if there exists at least one valid sequence of length k-1 ending before contest i.
                # Check if max_prev_val is effectively not negative infinity.
                # Using a threshold check with a large negative number for robustness with floating point comparisons.
                if max_prev_val > -1e18: # Check threshold instead of exact -inf
                    dp[i][k] = P[i-1] + 0.9 * max_prev_val
                # If max_prev_val is -inf, dp[i][k] remains at its initialized value of -inf,
                # correctly indicating that no sequence of length k ending at i is possible through this path.
            
            # After computing dp[i][k], check if it represents a valid state (not -inf).
            # If valid, calculate the rating corresponding to this state and update the overall maximum rating.
            if dp[i][k] > -1e18: # Check if dp[i][k] is a valid finite value
                 # Calculate the rating using the given formula:
                 # Rating = (Numerator / Denominator) - Penalty term
                 # Numerator sum is dp[i][k]
                 # Denominator (sum of weights) is W[k]
                 # Penalty term is 1200 / sqrt(k)
                 current_rating = dp[i][k] / W[k] - 1200.0 / math.sqrt(k)
                 
                 # Update the maximum rating found across all considered sequence lengths and ending contests.
                 max_R = max(max_R, current_rating)

    # Print the final maximum rating found.
    # Format the output to high precision (e.g., 17 decimal places) to ensure it meets 
    # the problem's required absolute or relative error tolerance of 10^-6.
    print(f"{max_R:.17f}")

# Execute the solve function when the script runs
solve()