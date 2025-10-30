import sys
import math
import itertools

def solve():
    """
    An expert Python programmer's solution to the printing machine problem.

    This function reads the problem input, calculates the minimum time required to
    print all line segments, and prints the result to standard output with the
    required precision.

    The method combines exhaustive search over permutations with dynamic programming
    to efficiently solve this TSP-variant.
    """
    try:
        # Read problem parameters and segment coordinates from standard input
        input_line = sys.stdin.readline()
        if not input_line.strip(): return
        N, S, T = map(int, input_line.split())
        segments = []
        for _ in range(N):
            A, B, C, D = map(int, sys.stdin.readline().split())
            segments.append(((A, B), (C, D)))
    except (IOError, ValueError):
        # Handle potential empty lines or malformed input
        return

    def distance(p1, p2):
        """Calculates the Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # --- Step 1: Calculate the constant printing time ---
    # The total time spent printing is the sum of all segment lengths divided by
    # the printing speed T. This value is constant regardless of the printing order.
    total_print_dist = 0
    for i in range(N):
        p1, p2 = segments[i]
        total_print_dist += distance(p1, p2)
    total_print_time = total_print_dist / T

    if N == 0:
        print(f"{0.0:.10f}")
        return

    # --- Step 2: Find the minimum possible travel distance using DP ---
    # We iterate through all N! permutations of segments. For each permutation,
    # we use dynamic programming to find the minimum travel distance.
    min_overall_travel_dist = float('inf')
    
    indices = list(range(N))
    
    for p in itertools.permutations(indices):
        # p is a tuple representing the order of segments, e.g., (1, 0, 2).
        
        # DP table for this permutation:
        # dp[i][j] = min travel dist for first i segments of p, ending at endpoint j of segment p[i-1].
        # i is 1-based, j is 0 or 1.
        dp = [[0.0, 0.0] for _ in range(N + 1)]

        # Base case (i=1): Travel from origin (0,0) to the first segment's start point.
        seg_idx_0 = p[0]
        ep0_0, ep0_1 = segments[seg_idx_0]
        origin = (0, 0)
        
        # To end at ep0_0 (endpoint 0), we must print from ep0_1. Travel: origin -> ep0_1.
        dp[1][0] = distance(origin, ep0_1)
        # To end at ep0_1 (endpoint 1), we must print from ep0_0. Travel: origin -> ep0_0.
        dp[1][1] = distance(origin, ep0_0)
        
        # DP for the rest of the segments (i from 2 to N)
        for i in range(2, N + 1):
            prev_seg_idx = p[i - 2]
            prev_ep0, prev_ep1 = segments[prev_seg_idx]
            
            curr_seg_idx = p[i - 1]
            curr_ep0, curr_ep1 = segments[curr_seg_idx]
            
            # Calculate dp[i][0]: end at curr_ep0 (print from curr_ep1)
            # Travel from the end of the previous segment to curr_ep1.
            dist_from_prev0 = dp[i - 1][0] + distance(prev_ep0, curr_ep1)
            dist_from_prev1 = dp[i - 1][1] + distance(prev_ep1, curr_ep1)
            dp[i][0] = min(dist_from_prev0, dist_from_prev1)
            
            # Calculate dp[i][1]: end at curr_ep1 (print from curr_ep0)
            # Travel from the end of the previous segment to curr_ep0.
            dist_from_prev0 = dp[i - 1][0] + distance(prev_ep0, curr_ep0)
            dist_from_prev1 = dp[i - 1][1] + distance(prev_ep1, curr_ep0)
            dp[i][1] = min(dist_from_prev0, dist_from_prev1)

        # Minimum travel distance for this specific permutation
        min_perm_travel_dist = min(dp[N][0], dp[N][1])
        
        # Update the overall minimum travel distance
        min_overall_travel_dist = min(min_overall_travel_dist, min_perm_travel_dist)

    # --- Step 3: Calculate and print the final total time ---
    min_travel_time = min_overall_travel_dist / S
    total_min_time = total_print_time + min_travel_time
    
    print(f"{total_min_time:.10f}")

solve()