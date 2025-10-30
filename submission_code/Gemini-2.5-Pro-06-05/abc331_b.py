# YOUR CODE HERE
import sys

def solve():
    """
    Reads input, solves the egg pack problem, and prints the answer.
    """
    # Read the input values for N, S, M, and L from a single line.
    try:
        N, S, M, L = map(int, sys.stdin.readline().split())
    except (IOError, ValueError):
        # This handles cases like empty input at the end of a file.
        return

    # We use dynamic programming to find the minimum cost.
    # Let dp[i] be the minimum cost to buy EXACTLY i eggs.
    # The size of our DP table needs to be large enough to account for "overshooting" N.
    # The largest pack is 12, so we might need to calculate up to N+11 eggs.
    # A size of N + 12 is safe.
    MAX_EGGS = N + 12

    # Initialize the DP table. The cost for 0 eggs is 0.
    # All other costs are initialized to infinity.
    dp = [float('inf')] * MAX_EGGS
    dp[0] = 0

    # Fill the DP table by iterating from 1 up to the maximum number of eggs.
    for i in range(1, MAX_EGGS):
        # Option 1: Arrive at `i` eggs by adding a 6-pack.
        if i >= 6:
            dp[i] = min(dp[i], dp[i - 6] + S)
    
        # Option 2: Arrive at `i` eggs by adding an 8-pack.
        if i >= 8:
            dp[i] = min(dp[i], dp[i - 8] + M)
        
        # Option 3: Arrive at `i` eggs by adding a 12-pack.
        if i >= 12:
            dp[i] = min(dp[i], dp[i - 12] + L)

    # The problem asks for the minimum cost for AT LEAST N eggs.
    # This means we must check the costs for N, N+1, N+2, ... eggs and take the minimum.
    # Our DP table up to MAX_EGGS covers all relevant possibilities.
    result = min(dp[N:])

    # Print the final minimum cost to standard output.
    print(result)

solve()