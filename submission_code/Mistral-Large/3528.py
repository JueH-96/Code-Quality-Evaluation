from typing import List

class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 0

        dp = [0] * n
        max_score = 0
        max_index = 0

        for i in range(1, n):
            dp[i] = max_score + i * nums[max_index] - max_index * nums[max_index]
            if dp[i] > max_score + (i - max_index) * nums[max_index]:
                max_score = dp[i]
                max_index = i

        return dp[n-1]