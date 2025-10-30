import collections
from typing import List

class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        n = len(nums)
        MOD = 10**9 + 7

        # dp[current_sum][current_length]
        # dp[s][l] will store the number of subsequences from nums that sum to 's' and have length 'l'.
        # The sum 's' can range from 0 to k.
        # The length 'l' can range from 0 to n.
        # Initialize dp table with zeros.
        dp = [[0] * (n + 1) for _ in range(k + 1)]

        # Base case: There is one way to form an empty subsequence (sum 0, length 0).
        dp[0][0] = 1

        # Iterate through each number in the input array nums.
        for num in nums:
            # When processing a new 'num', iterate 'current_sum' and 'current_length' downwards.
            # This ensures that each 'num' is used at most once for the subsequences counted
            # within this iteration's updates.
            # Iterate 'current_sum' from k down to 'num'.
            # If current_sum < num, 'num' cannot be included to form 'current_sum'
            # (as nums[i] >= 1, so no negative values).
            for current_sum in range(k, num - 1, -1):
                # Iterate 'current_length' from n down to 1.
                # A subsequence of length 0 cannot become length 1 by adding 'num'
                # if we are calculating based on 'current_length - 1'.
                # We need at least one previous element to add 'num' to.
                for current_length in range(n, 0, -1):
                    # To form a subsequence of sum 'current_sum' and length 'current_length'
                    # by including the current 'num':
                    # We must have previously formed a subsequence of sum 'current_sum - num'
                    # and length 'current_length - 1'.
                    # Add the count of such previous subsequences to dp[current_sum][current_length].
                    dp[current_sum][current_length] = (dp[current_sum][current_length] + 
                                                      dp[current_sum - num][current_length - 1]) % MOD
        
        total_power_sum = 0

        # After filling the DP table, calculate the total sum of powers according to the derived formula:
        # Sum over all subsequences S_k of nums (that sum to k) of (2^(n - |S_k|)).
        for length in range(1, n + 1):
            # Get the number of subsequences from nums that sum to 'k' and have 'length'.
            num_subsequences_sum_k_length_l = dp[k][length]

            # If there are such subsequences:
            if num_subsequences_sum_k_length_l > 0:
                # For each such subsequence S_k of length 'length':
                # It can be extended to a "super-subsequence" S by including any combination
                # of the remaining (n - length) elements from the original 'nums' array.
                # There are 2^(n - length) ways to choose these remaining elements.
                # Each of these 2^(n - length) super-subsequences S will have S_k as one of its
                # own subsequences summing to k, thus contributing 1 to S's power.
                # So, S_k effectively contributes 2^(n - length) to the total sum of powers.
                
                # Calculate 2^(n - length) modulo MOD.
                ways_to_extend = pow(2, n - length, MOD)
                
                # Add (count of S_k * ways_to_extend) to the total sum.
                total_power_sum = (total_power_sum + 
                                   num_subsequences_sum_k_length_l * ways_to_extend) % MOD
        
        return total_power_sum