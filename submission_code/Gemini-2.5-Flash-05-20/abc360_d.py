import sys
import bisect

def solve():
    N, T = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    X = list(map(int, sys.stdin.readline().split()))

    # Separate ants into right-movers and left-movers based on their initial coordinates.
    # We only need their coordinates for the bisect search.
    right_coords = []
    left_coords = []

    for i in range(N):
        if S[i] == '1': # Facing positive direction (right)
            right_coords.append(X[i])
        else: # Facing negative direction (left)
            left_coords.append(X[i])

    # Sort the coordinates for efficient searching using bisect.
    right_coords.sort()
    left_coords.sort()

    total_passing_pairs = 0
    
    # The maximum distance two ants can be apart to pass each other within time T+0.1
    # is 2 * T.
    # This is derived from:
    # 1. t_pass = |X_i - X_j| / 2
    # 2. t_pass < T + 0.1
    # Combining these: |X_i - X_j| / 2 < T + 0.1
    #                 |X_i - X_j| < 2T + 0.2
    # Since X_i and X_j are integers, |X_i - X_j| is an integer.
    # For an integer K, K < SomeInteger + 0.2 is equivalent to K <= SomeInteger.
    # So, |X_i - X_j| <= 2T.
    max_passing_distance = 2 * T

    # Iterate through each right-moving ant
    for r_x in right_coords:
        # A right-moving ant at r_x passes a left-moving ant at l_x if:
        # 1. l_x > r_x (they are positioned to move towards each other)
        # 2. l_x - r_x <= max_passing_distance (they meet within the time limit)
        # Combining these, we are looking for l_x such that r_x < l_x <= r_x + max_passing_distance.

        # Find the index of the first left-moving ant coordinate >= r_x + 1
        # This determines the lower bound of the search range for l_x.
        lower_bound_idx = bisect.bisect_left(left_coords, r_x + 1)
        
        # Find the index of the first left-moving ant coordinate > r_x + max_passing_distance
        # This determines the upper bound of the search range for l_x.
        # bisect_right returns an insertion point which comes after (or at) any existing entries of the value.
        # So, elements up to index (upper_bound_idx - 1) will be <= r_x + max_passing_distance.
        upper_bound_idx = bisect.bisect_right(left_coords, r_x + max_passing_distance)
        
        # The number of left-moving ants in the valid range is the difference of these indices.
        total_passing_pairs += (upper_bound_idx - lower_bound_idx)

    sys.stdout.write(str(total_passing_pairs) + "
")

solve()