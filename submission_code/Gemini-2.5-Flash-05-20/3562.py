import bisect
from functools import cmp_to_key
from typing import List

class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        # Step 1: Augment intervals with their original indices
        # Each element will be (l, r, weight, original_index)
        indexed_intervals = []
        for i, (l, r, w) in enumerate(intervals):
            indexed_intervals.append((l, r, w, i))

        # Step 2: Sort intervals
        # The primary sort key for interval scheduling is the end time (r).
        # When end times are equal, sort by start time (l).
        # If start and end times are equal, sort by weight in descending order (to prioritize heavier intervals).
        # Finally, if all above are equal, sort by original index in ascending order.
        # This sorting helps in processing intervals in a DP-friendly order.
        intervals_sorted = sorted(indexed_intervals, key=lambda x: (x[1], x[0], -x[2], x[3]))

        N = len(intervals_sorted)
        MAX_K = 4 # Maximum number of intervals to choose

        # dp[k][i] will store a tuple (total_weight, list_of_indices)
        # total_weight: sum of weights of chosen intervals.
        # list_of_indices: a sorted list of original indices of chosen intervals.
        # dp[k][i] represents the best choice using AT MOST 'k' intervals from the first 'i'
        # intervals_sorted (i.e., intervals_sorted[0...i-1]).
        
        # Initialize dp table.
        # dp[0][...] should be (0, []) for any i, representing choosing 0 intervals.
        # dp[k][0] for k > 0 should be (0, []), representing choosing from an empty set.
        dp = [[(0, []) for _ in range(N + 1)] for _ in range(MAX_K + 1)]

        # Helper function for custom comparison based on problem rules
        # Returns True if res1 is "better" than res2 (higher weight, or lexicographically smaller indices if weights are equal)
        def is_better(res1, res2):
            weight1, indices1 = res1
            weight2, indices2 = res2

            if weight1 > weight2:
                return True
            if weight1 < weight2:
                return False
            
            # Weights are equal, compare indices lexicographically
            # Python's list comparison naturally works lexicographically
            return indices1 < indices2

        # Precompute a list of end times from intervals_sorted for efficient binary search
        # This list's indices correspond directly to intervals_sorted's indices.
        end_times_sorted = [iv[1] for iv in intervals_sorted]

        # Step 3: Fill DP table
        # Iterate through each interval in the sorted list
        for i in range(1, N + 1): # i corresponds to intervals_sorted[i-1]
            l_curr, r_curr, w_curr, original_idx_curr = intervals_sorted[i-1]

            # For each possible number of chosen intervals (from 1 to MAX_K)
            for k in range(1, MAX_K + 1):
                # Option 1: Do not include the current interval (intervals_sorted[i-1])
                # In this case, the best result is simply the best result considering intervals up to (i-1) with k intervals.
                res_option1 = dp[k][i-1]

                # Option 2: Include the current interval (intervals_sorted[i-1])
                # To include the current interval, we must find a set of (k-1) non-overlapping intervals
                # that end before l_curr.
                
                # Find the largest index 'p_idx' in intervals_sorted such that intervals_sorted[p_idx-1] ends
                # strictly before l_curr (i.e., intervals_sorted[p_idx-1][1] < l_curr).
                # bisect_left returns an insertion point 'x' such that all elements in the slice `array[:x]`
                # are less than the value being searched for.
                # So, `p_idx = bisect_left(end_times_sorted, l_curr)` means that all intervals
                # `intervals_sorted[0...p_idx-1]` have end times strictly less than `l_curr`.
                # We can then take the best `k-1` intervals from this prefix `intervals_sorted[0...p_idx-1]`,
                # which is stored in `dp[k-1][p_idx]`.
                
                p_idx = bisect.bisect_left(end_times_sorted, l_curr)
                
                prev_weight, prev_indices = dp[k-1][p_idx]
                
                # Construct the new list of indices by adding the current interval's original index and sorting it.
                # Sorting here is crucial for the lexicographical comparison.
                current_indices_option2 = sorted(prev_indices + [original_idx_curr])
                current_weight_option2 = prev_weight + w_curr
                res_option2 = (current_weight_option2, current_indices_option2)

                # Choose the better result between Option 1 and Option 2
                if is_better(res_option2, res_option1):
                    dp[k][i] = res_option2
                else:
                    dp[k][i] = res_option1
        
        # Step 4: Find the overall best result from the entire DP table
        # Iterate through all possible counts of intervals (1 to MAX_K) considering all intervals (N)
        best_overall_result = (0, []) # Initialize with a base case of 0 weight and empty indices

        for k in range(1, MAX_K + 1):
            if is_better(dp[k][N], best_overall_result):
                best_overall_result = dp[k][N]
        
        # Return the list of original indices from the best result found
        return best_overall_result[1]