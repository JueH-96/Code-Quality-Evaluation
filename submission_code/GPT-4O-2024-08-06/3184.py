from typing import List

class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        # dp[i] will store the maximum sum of a balanced subsequence ending at index i
        dp = nums[:]  # Initialize dp with nums because a single element is a balanced subsequence

        # Iterate over each element in nums
        for i in range(n):
            # Check all previous elements to see if they can form a balanced subsequence with nums[i]
            for j in range(i):
                if nums[i] - nums[j] >= i - j:
                    # If the condition is satisfied, update dp[i] if we get a better sum
                    dp[i] = max(dp[i], dp[j] + nums[i])

        # The result is the maximum value in dp, which represents the maximum sum of a balanced subsequence
        return max(dp)