class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [1] * n
        for i in range(1, n):
            for j in range(i):
                if sum(nums[j:i+1]) >= nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)