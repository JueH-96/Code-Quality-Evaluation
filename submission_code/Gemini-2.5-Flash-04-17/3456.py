from typing import List
import collections

class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        # Map distinct values to indices
        # Using a set then sorting helps handle large value ranges efficiently.
        distinct_values = sorted(list(set(nums)))
        value_to_idx = {v: i for i, v in enumerate(distinct_values)}
        m = len(distinct_values)

        # dp[d][v_idx] = max length of a good subsequence ending with distinct_values[v_idx]
        # with exactly d adjacent differences, using elements processed so far (up to current index).
        # Initialize with 0.
        dp = [[0] * m for _ in range(k + 1)]

        max_overall_len = 0

        # Iterate through each number in the input array
        for num in nums:
            current_val_idx = value_to_idx[num]

            # Calculate potential maximum lengths for subsequences that *must end* with the current number
            # (at the current index in the original array) and have exactly d differences.
            # max_len_ending_current_val[d] stores this.
            # Initialize with 1 for 0 differences (the current number itself forms a length-1 sequence)
            max_len_ending_current_val = [0] * (k + 1)
            max_len_ending_current_val[0] = 1 # The subsequence consists only of the current number [num]

            # Consider extending existing subsequences ending before the current index (i.e., from dp table)
            # The dp table currently holds maximum lengths ending at previous indices.
            for prev_diff in range(k + 1):
                for prev_v_idx in range(m):
                    # Consider a subsequence ending with values[prev_v_idx] with exactly prev_diff differences
                    # using elements processed before the current index. Its max length is dp[prev_diff][prev_v_idx].
                    if dp[prev_diff][prev_v_idx] > 0: # Only consider valid existing subsequences
                        current_length = dp[prev_diff][prev_v_idx]

                        # Case 1: Extend sequence ending with the same value
                        if current_val_idx == prev_v_idx:
                            new_diff = prev_diff
                            new_length = current_length + 1
                            # Update the potential length ending with current_val and new_diff differences
                            max_len_ending_current_val[new_diff] = max(max_len_ending_current_val[new_diff], new_length)

                        # Case 2: Extend sequence ending with a different value
                        else:
                            new_diff = prev_diff + 1
                            if new_diff <= k:
                                new_length = current_length + 1
                                # Update the potential length ending with current_val and new_diff differences
                                max_len_ending_current_val[new_diff] = max(max_len_ending_current_val[new_diff], new_length)

            # Update the dp table for the current value index.
            # The maximum length of a subsequence ending with the current value (distinct_values[current_val_idx])
            # and exactly d differences, considering elements up to the current index,
            # is the value computed in max_len_ending_current_val[d]. This effectively replaces
            # the previous max length ending with this value at an earlier index.
            for d in range(k + 1):
                 dp[d][current_val_idx] = max_len_ending_current_val[d]

            # Update the overall maximum length found so far.
            # This is the maximum value among all max_len_ending_current_val[d] for the current number,
            # as these represent the longest sequences ending at the current position.
            max_overall_len = max(max_overall_len, max(max_len_ending_current_val))

        # The final answer is the maximum length among all good subsequences found.
        # A good subsequence has at most k differences. Our DP state stores
        # max lengths for *exactly* d differences, where d goes up to k.
        # The max_overall_len tracks the maximum length found across all ending values and all
        # difference counts <= k throughout the process.
        return max_overall_len