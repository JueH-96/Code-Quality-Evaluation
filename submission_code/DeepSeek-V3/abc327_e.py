# YOUR CODE HERE
import sys
import math

def main():
    N, *rest = map(int, sys.stdin.read().split())
    P = rest[:N]
    
    # Precompute the weights for each possible k
    # We need to find the best subset for each possible k, and then choose the maximum R
    # Since N is up to 5000, we need an efficient way to compute the best subset for each k
    
    # We will use dynamic programming to compute the best subset for each k
    # Let dp[k][i] be the maximum sum of (0.9)^(k-j) * P_j for the first i contests, choosing exactly k contests
    
    # Initialize dp
    # dp[k][i] = max(dp[k-1][j] + (0.9)^(k-1) * P_i for j < i)
    
    # To optimize, we can precompute the powers of 0.9
    power = [1.0]
    for i in range(1, N+1):
        power.append(power[-1] * 0.9)
    
    # Initialize dp[k][i] as a list of lists
    # Since k can be up to N, and i up to N, we need to manage memory efficiently
    # We will use a list of lists, but for each k, we only need the previous k-1's dp
    
    # Initialize dp[0][i] = 0 for all i
    # dp[1][i] = P_i * power[0] = P_i * 1.0
    # For k > 1, dp[k][i] = max(dp[k-1][j] + P_i * power[k-1] for j < i)
    
    # To optimize, we can compute the dp in a way that for each k, we iterate through i and keep track of the maximum dp[k-1][j] for j < i
    
    # Initialize dp as a list of dictionaries or lists
    # Since N is up to 5000, a list of lists is manageable
    
    # Initialize dp[k][i] as a list of lists
    # dp[k][i] will store the maximum sum for the first i contests, choosing exactly k contests
    
    # Initialize dp[0][i] = 0 for all i
    # dp[1][i] = P_i * power[0] = P_i * 1.0
    # For k > 1, dp[k][i] = max(dp[k-1][j] + P_i * power[k-1] for j < i)
    
    # To manage memory, we will compute dp[k][i] for each k in order, and for each k, we will keep track of the maximum dp[k-1][j] for j < i
    
    # Initialize dp as a list of lists
    # dp[k] will be a list of size N+1, initialized to 0
    # We will compute dp[k] based on dp[k-1]
    
    # Initialize dp[0] as all zeros
    dp_prev = [0.0] * (N+1)
    
    # Initialize the maximum R
    max_R = -float('inf')
    
    # Iterate over k from 1 to N
    for k in range(1, N+1):
        dp_current = [0.0] * (N+1)
        max_prev = 0.0
        for i in range(1, N+1):
            # dp_current[i] = max(dp_prev[j] + P[i-1] * power[k-1] for j < i)
            # To optimize, we can keep track of the maximum dp_prev[j] for j < i
            dp_current[i] = max_prev + P[i-1] * power[k-1]
            # Update max_prev for the next i
            if dp_prev[i] > max_prev:
                max_prev = dp_prev[i]
        # Now, compute the sum of weights for k contests
        sum_weights = sum(power[i] for i in range(k))
        # Compute the rating R for this k
        R = (dp_current[N] / sum_weights) - (1200.0 / math.sqrt(k))
        if R > max_R:
            max_R = R
        # Update dp_prev for the next k
        dp_prev = dp_current
    
    # Print the maximum R with 15 decimal places
    print("{0:.15f}".format(max_R))

if __name__ == "__main__":
    main()