from collections import Counter
from typing import List

class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        MOD = 10**9 + 7

        # 1. Count frequencies
        freq = Counter(nums)
        count_zero = freq.get(0, 0)
        
        # Remove 0 from frequencies as it's handled separately later
        if 0 in freq:
            del freq[0]

        # 2. Initialize DP table
        # dp[i] will store the number of ways to achieve sum i using non-zero elements
        # We only need sums up to r.
        max_sum_dp = r
        dp = [0] * (max_sum_dp + 1)
        dp[0] = 1 # Empty multiset sum is 0 (using only non-zero elements)

        # 3. Process distinct non-zero values <= r
        # Values > r cannot be part of a sum <= r, so we can ignore them.
        distinct_positive_values = sorted([v for v in freq.keys() if v <= max_sum_dp])

        for v in distinct_positive_values:
            c = freq[v]
            
            # We use the sliding window optimization for bounded knapsack DP
            # Iterate through residues `rem` modulo `v`
            for rem in range(v):
                # Collect the current dp values for sums j = rem, rem+v, rem+2v, ... up to max_sum_dp
                # This represents the 'row' of the DP table for this residue before processing `v`
                indices_in_rem = []
                temp_vals = []
                for j in range(rem, max_sum_dp + 1, v):
                    indices_in_rem.append(j)
                    temp_vals.append(dp[j])
                
                # If there are no sums for this residue <= max_sum_dp, continue
                if not indices_in_rem:
                    continue

                # Use sliding window sum on temp_vals (which are old dp values)
                current_sum = 0
                # The window size is c + 1 (representing using v 0 to c times)
                window_size = c + 1

                # Iterate through the sums j = rem + i*v within this residue class
                # i is the index within temp_vals and indices_in_rem
                for i in range(len(indices_in_rem)):
                    # Add the current value (old dp value for indices_in_rem[i]) into the window sum
                    current_sum = (current_sum + temp_vals[i]) % MOD

                    # If the window is now conceptually larger than c+1 elements, remove the oldest element
                    # The oldest element corresponds to using v more than c times (specifically c+1 times the step v)
                    # The window contains temp_vals[i], temp_vals[i-1], ..., temp_vals[i-window_size + 1]
                    # If i >= window_size, the element temp_vals[i - window_size] is falling out of the window
                    if i >= window_size:
                        val_to_remove_idx = i - window_size
                        val_to_remove = temp_vals[val_to_remove_idx]
                        current_sum = (current_sum - val_to_remove + MOD) % MOD # Add MOD to handle potential negative result

                    # The current_sum calculated here is the new_dp value for the sum indices_in_rem[i]
                    dp[indices_in_rem[i]] = current_sum

        # 5. Handle the zero value
        # Any sub-multiset formed using non-zero elements can be combined with
        # 0, 1, ..., count_zero zeros. This multiplies the count for each sum S
        # achieved by non-zero elements by (count_zero + 1).
        # This applies to the sum 0 as well (the empty set combination).
        multiplier = (count_zero + 1) % MOD
        for i in range(max_sum_dp + 1):
            dp[i] = (dp[i] * multiplier) % MOD

        # 6. Sum up dp values in the range [l, r]
        result = 0
        # The DP table dp[i] now correctly counts sub-multisets (including zeros)
        # that sum up to i.
        for i in range(l, r + 1):
            result = (result + dp[i]) % MOD

        return result