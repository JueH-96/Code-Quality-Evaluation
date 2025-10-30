# YOUR CODE HERE
import sys
import math

def solve():
    """
    Reads input, solves the problem, and prints the answer.
    """
    # Read problem input using fast I/O
    try:
        N = int(sys.stdin.readline())
        points = [tuple(map(int, sys.stdin.readline().split())) for _ in range(N)]
    except (IOError, ValueError):
        # Exit gracefully on input errors, though not expected in a contest environment.
        return

    # A safe upper bound for the optimal number of skips, based on the exponential
    # growth of the penalty function.
    C_max = 40

    # dp[i][c]: minimum distance to reach checkpoint i (0-indexed) with c skips.
    dp = [[float('inf')] * (C_max + 1) for _ in range(N)]

    # Base case: at checkpoint 0, distance is 0, skips are 0.
    dp[0][0] = 0.0

    # Helper function for Euclidean distance
    def get_dist(p1_idx, p2_idx):
        p1, p2 = points[p1_idx], points[p2_idx]
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # Dynamic Programming Calculation
    # Time complexity: O(N * C_max^2)
    for i in range(1, N):  # Current checkpoint index, from 1 to N-1
        # c is the total number of checkpoints skipped to reach checkpoint i.
        # Max skips to reach i is i-1 (by path 0 -> i).
        for c in range(min(i, C_max + 1)):
            # p is the previous checkpoint's index.
            # Number of skips between p and i is (i - p - 1).
            # So, number of skips to reach p must be c_prev = c - (i - p - 1).
            # For c_prev to be non-negative, p must be >= i - c - 1.
            for p in range(max(0, i - c - 1), i):
                skips_now = i - p - 1
                c_prev = c - skips_now
                
                # Check if the previous state was reachable
                if dp[p][c_prev] != float('inf'):
                    distance_p_i = get_dist(p, i)
                    dp[i][c] = min(dp[i][c], dp[p][c_prev] + distance_p_i)

    # Calculate the final minimum total cost 's'
    min_s = float('inf')

    # Iterate through all possible numbers of skips c to reach the final checkpoint.
    # The max possible skips is N-2. We check up to our bound C_max.
    for c in range(min(N - 1, C_max + 1)):
        # If this state is reachable
        if dp[N - 1][c] != float('inf'):
            # Calculate penalty
            penalty = 0
            if c > 0:
                penalty = 1 << (c - 1)  # 2**(c-1)
            
            total_cost = dp[N - 1][c] + penalty
            min_s = min(min_s, total_cost)

    # Print the result with the required precision
    print(f"{min_s:.15f}")

solve()