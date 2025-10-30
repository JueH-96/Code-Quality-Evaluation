import collections
from typing import List

class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        MOD = 10**9 + 7
        
        # Count frequencies of each number in nums
        counts = collections.Counter(nums)
        
        # Extract the count of zeros and remove it from the counts map
        # Zeros are handled separately as they don't change the sum but create distinct multisets.
        num_zeros = counts.get(0, 0)
        if 0 in counts:
            del counts[0]
        
        # Initialize dp array. dp[s] will store the number of ways to form sum s.
        # The size is r + 1 because we need to compute sums up to r.
        dp = [0] * (r + 1)
        
        # Base case: One way to form sum 0 (by choosing an empty multiset).
        dp[0] = 1
        
        # Iterate through each unique non-zero number and its frequency
        # The constraint "Sum of nums does not exceed 2 * 10^4" implies that the number
        # of distinct non-zero elements is relatively small (at most ~200).
        # This makes the O(r * U) complexity (where U is unique non-zero elements) efficient enough.
        for num, freq in counts.items():
            # If the current number is greater than the maximum target sum 'r',
            # it cannot contribute to any sum within [0, r].
            if num > r:
                continue

            # Create a copy of the dp array before processing the current 'num'.
            # This 'dp_old' is used for the correction step (subtracting overcounted combinations).
            dp_old = list(dp)

            # Iterate through possible sums from 'num' up to 'r'.
            # We iterate forwards because dp[j - num] should reflect the state
            # after potentially including 'num' for smaller sums (i.e., if we add another 'num').
            for j in range(num, r + 1):
                # Add ways to form sum (j - num) using available items (including 'num').
                # This treats 'num' as if it has infinite supply for this sum calculation.
                dp[j] = (dp[j] + dp[j - num]) % MOD

                # Correction step: If we have used 'num' more than its allowed frequency 'freq',
                # we must subtract those invalid combinations.
                # dp_old[j - (freq + 1) * num] represents ways to form sum `j - (freq + 1) * num`
                # using elements considered *before* this current 'num'.
                # When we add `(freq + 1)` copies of `num` to these previous combinations,
                # it creates an invalid sum `j` (because it uses more than `freq` copies of `num`).
                # We subtract these overcounted paths.
                if j >= (freq + 1) * num:
                    dp[j] = (dp[j] - dp_old[j - (freq + 1) * num] + MOD) % MOD
        
        # After processing all non-zero numbers, dp[s] contains the count of sub-multisets
        # summing to 's' using only non-zero elements.
        
        # Now, account for the zero values.
        # If there are 'num_zeros' copies of 0, we can choose to include 0, 1, ..., num_zeros of them.
        # This gives (num_zeros + 1) distinct ways to pick zeros (e.g., {}, {0}, {0,0}, ...).
        # Since adding zeros does not change the sum, each way to achieve sum 's' using non-zero
        # elements can be combined with any of these (num_zeros + 1) ways to pick zeros.
        # This means we multiply each dp[s] by (num_zeros + 1).
        # This also correctly handles dp[0]: (1 * (num_zeros + 1)) results in (num_zeros + 1)
        # ways to form sum 0 (empty set, one 0, two 0s, etc.).
        for s in range(r + 1):
            dp[s] = (dp[s] * (num_zeros + 1)) % MOD
            
        # Finally, sum up the counts for all sums within the desired range [l, r].
        total_count = 0
        for s in range(l, r + 1):
            total_count = (total_count + dp[s]) % MOD
            
        return total_count