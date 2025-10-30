from typing import List

class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        dp = [1] * 301  # dp[d] represents the max length with last difference d
        freq = [0] * 301  # freq[val] counts occurrences of each value
        
        for num in nums:
            # Compute suffix maximums for non-increasing differences
            suffix_max = [0] * 301
            suffix_max[300] = dp[300]
            for d in range(299, -1, -1):
                suffix_max[d] = max(dp[d], suffix_max[d + 1])
            
            # Create a copy of current dp to update for this step
            new_dp = dp.copy()
            
            # Update dp for all possible differences based on frequency
            for val in range(1, 301):
                if freq[val] > 0:
                    diff = abs(num - val)
                    if diff <= 300:
                        new_dp[diff] = max(new_dp[diff], suffix_max[diff] + 1)
            
            # Assign the updated dp
            dp = new_dp
            # Update frequency of the current number
            freq[num] += 1
        
        return max(dp)