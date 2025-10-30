from typing import List

class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        # Sort unique reward values in ascending order.
        # This helps process rewards in a way that aligns with the DP state definition.
        # Duplicates of the same value greater than the current sum offer no extra benefit
        # compared to having just one instance of that value because you can only pick a reward
        # if it's strictly greater than the current sum. If you have two rewards of value `r`,
        # and pick one when current sum is `x` (r > x), the new sum is `x+r`. To pick the second `r`,
        # you'd need `r > x+r`, which is impossible since `r > 0` and `x >= 0`.
        unique_rewards = sorted(list(set(rewardValues)))

        # Use a bitset (represented by a Python integer) to track achievable sums.
        # The i-th bit of `dp` is set if sum `i` is achievable.
        # Initially, only sum 0 is achievable.
        dp = 1 # Corresponds to dp[0] = True

        # Iterate through each unique reward value `r` in increasing order.
        # For each `r`, we update the set of achievable sums.
        for r in unique_rewards:
            # If a sum 's' is currently achievable (i.e., the 's'-th bit in `dp` is set),
            # and if `r > s`, then the sum `s + r` becomes achievable.
            # We need to apply this rule for all currently achievable sums 's' that are strictly less than 'r'.
            # In terms of bitset operations:
            # 1. We are interested in the subset of currently achievable sums 's' where `s < r`.
            #    The mask `(1 << r) - 1` has bits 0 through `r-1` set.
            #    `dp & ((1 << r) - 1)` filters `dp` to keep only bits `s` where `s < r`.
            # 2. For each such sum `s`, the sum `s + r` becomes achievable. Shifting the filtered bitset
            #    left by `r` (`<< r`) achieves this: if bit `s` was set in the filtered result,
            #    bit `s + r` will be set in the shifted result.
            # 3. The new set of achievable sums is the union of the original achievable sums (`dp`)
            #    and the new sums (`s + r` for `s < r`). The bitwise OR operation (`|`) performs this union.
            
            # Calculate the mask for sums strictly less than r.
            # If r = 1, mask = (1 << 1) - 1 = 1 (binary 01), filters sum 0.
            # If r = 2, mask = (1 << 2) - 1 = 3 (binary 11), filters sums 0, 1.
            # If r = 3, mask = (1 << 3) - 1 = 7 (binary 111), filters sums 0, 1, 2.
            mask = (1 << r) - 1
            
            # Filter achievable sums s such that s < r and shift them by r.
            newly_achievable_sums = (dp & mask) << r
            
            # Combine the existing achievable sums with the newly achievable sums.
            dp = dp | newly_achievable_sums

        # The maximum total reward is the largest achievable sum.
        # This corresponds to the index of the highest set bit in the `dp` integer.
        # The `bit_length()` method returns the number of bits required to represent the integer `dp`.
        # This is equivalent to the position of the most significant (highest) set bit plus one.
        # For example, if dp = 0b11011 (decimal 27), bit_length() is 5. The bits are at indices 0, 1, 3, 4.
        # The highest set bit is at index 4. `dp.bit_length() - 1 = 5 - 1 = 4`.
        # If dp is 1 (only sum 0 achievable), bit_length() is 1, index is 0. This is correct.
        # Since constraints guarantee 1 <= rewardValues[i], unique_rewards is non-empty and contains at least one reward >= 1.
        # After processing the first unique reward `r` (where r >= 1), `dp` will become `1 | (1 << r)`,
        # which has at least bits 0 and `r` set, and thus `dp > 1`.
        # So, `dp.bit_length()` will be >= 2, and `dp.bit_length() - 1` will be >= 1.
        max_sum = dp.bit_length() - 1

        return max_sum