import collections
from typing import List

class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        # dp_val_breaks[value][breaks] stores the maximum length of a good subsequence
        # ending with 'value' and using 'breaks' differences.
        # Since 'value' can be large (up to 10^9), we use a defaultdict for 'value' keys.
        # Each value maps to a list of size (k+1) to store lengths for break counts 0 to k.
        dp_val_breaks = collections.defaultdict(lambda: [0] * (k + 1))

        # max_overall_len_for_breaks[b] stores the maximum length of any good subsequence
        # found so far that uses exactly 'b' differences, regardless of its ending value.
        max_overall_len_for_breaks = [0] * (k + 1)

        # Initialize overall_max_length. It will be at least 1 if nums is not empty,
        # as any single element forms a good subsequence of length 1 with 0 breaks.
        overall_max_length = 0

        # Iterate through each number in the input array nums
        for current_num in nums:
            # current_num_updates is a temporary list to store the calculated maximum lengths
            # for subsequences ending with `current_num` for each break count `b`.
            # This is crucial to ensure that updates for `current_num` are based on the DP state
            # *before* `current_num` itself starts influencing the `max_overall_len_for_breaks` for this iteration.
            current_num_updates = [0] * (k + 1)

            # Iterate through possible break counts from 0 to k
            for b in range(k + 1):
                # Option 1: `current_num` extends a subsequence that *also* ended with `current_num`.
                # This means no new "break" is introduced. The break count `b` remains the same.
                # If `current_num` is encountered for the first time for this `b`,
                # `dp_val_breaks[current_num][b]` will be 0 by default, so `len_if_same` will be 1 (for `current_num` itself).
                len_if_same = dp_val_breaks[current_num][b] + 1

                # Option 2: `current_num` extends a subsequence that ended with a *different* value.
                # This introduces one new "break". So, the previous subsequence must have used `b-1` breaks.
                len_if_diff = 0
                if b > 0: # Check if there is budget for a new break
                    # We take the maximum length achieved by *any* subsequence (ending with any value)
                    # that used `b-1` breaks, and then append `current_num`.
                    len_if_diff = max_overall_len_for_breaks[b - 1] + 1
                
                # The maximum length of a good subsequence ending with `current_num` and using `b` breaks
                # is the maximum of these two options.
                current_num_updates[b] = max(len_if_same, len_if_diff)
            
            # After computing all potential updates for `current_num` across all break counts,
            # we now commit these updates to the main DP tables.
            for b in range(k + 1):
                # Update the specific DP state for `current_num` and break count `b`.
                # `current_num_updates[b]` already holds the maximum length achieved for this specific combination.
                dp_val_breaks[current_num][b] = current_num_updates[b]
                
                # Update the overall maximum length found for any subsequence using `b` breaks.
                # This is important for "Option 2" in subsequent iterations.
                max_overall_len_for_breaks[b] = max(max_overall_len_for_breaks[b], dp_val_breaks[current_num][b])
            
            # After processing `current_num` and updating all `max_overall_len_for_breaks`,
            # update the global maximum length found so far.
            overall_max_length = max(overall_max_length, max(max_overall_len_for_breaks))

        return overall_max_length