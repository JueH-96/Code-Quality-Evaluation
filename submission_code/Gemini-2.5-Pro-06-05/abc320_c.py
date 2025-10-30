import sys
from collections import Counter
from itertools import product

# YOUR CODE HERE
def solve():
    """
    Solves the Slot Machine problem.
    """
    # Read problem inputs from standard input
    try:
        M = int(sys.stdin.readline())
        reels = [sys.stdin.readline().strip() for _ in range(3)]
    except (IOError, ValueError):
        # Handle potential empty input or invalid format
        print(-1)
        return

    # Initialize the minimum time to a very large value (infinity)
    min_total_time = float('inf')

    # We need to find a common digit 'd' and three stop times t1, t2, t3
    # such that S1[t1%M]==d, S2[t2%M]==d, S3[t3%M]==d and t1,t2,t3 are distinct.
    # The goal is to minimize max(t1, t2, t3).

    # Let's iterate through each possible target digit 'd' from '0' to '9'.
    for digit in "0123456789":
        
        # For the current digit, find all possible stop indices for each reel.
        # An index `j` is possible for reel `i` if S_i[j] == digit.
        positions_for_digit = []
        is_possible_for_digit = True
        for reel_str in reels:
            indices = [i for i, char in enumerate(reel_str) if char == digit]
            # If a reel doesn't have the current digit, it's impossible to
            # make all reels show this digit.
            if not indices:
                is_possible_for_digit = False
                break
            positions_for_digit.append(indices)

        # If this digit doesn't appear on all three reels, skip to the next digit.
        if not is_possible_for_digit:
            continue

        # Now, for the current possible digit, we iterate through all combinations
        # of stop indices, taking one index from each reel's list of possibilities.
        # For example, a combination could be (index_for_reel1, index_for_reel2, index_for_reel3).
        for index_combo in product(*positions_for_digit):
            # `index_combo` is a tuple like (j1, j2, j3) where S1[j1], S2[j2], S3[j3] are all equal to `digit`.
            # We need to find the minimum time to stop the reels at these indices.
            # The stop times t1, t2, t3 must be distinct.
            
            # We count the occurrences of each index in the combination.
            # e.g., for (8, 0, 8), counts would be {8: 2, 0: 1}.
            counts = Counter(index_combo)
            
            # Calculate the time for this specific combination.
            # If an index `j` appears `k` times in the combo (meaning `k` different reels
            # need to be stopped at index `j`), the reels must be stopped at times
            # congruent to `j` mod `M`. The earliest `k` distinct times for this are:
            # j, j+M, j+2M, ..., j+(k-1)*M.
            # The latest of these is `j + (k-1)*M`.
            # The total time for the `index_combo` is the maximum of these "latest times"
            # over all unique indices in the combo.
            combo_time = 0
            for index, num_occurrences in counts.items():
                # Calculate the time required for the last reel stopping at this index `j`.
                time_for_index = index + (num_occurrences - 1) * M
                # The total time for the combo is the max of times for each index group.
                combo_time = max(combo_time, time_for_index)
            
            # Update the overall minimum time found so far.
            min_total_time = min(min_total_time, combo_time)

    # If `min_total_time` was never updated from infinity, it means no solution
    # was found (no digit appeared on all three reels).
    if min_total_time == float('inf'):
        print(-1)
    else:
        print(min_total_time)

solve()