class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 0
        dp = [-float('inf')] * n
        dp[0] = 0
        for i in range(1, n):
            for j in range(i):
                if dp[j] != -float('inf'):
                    dp[i] = max(dp[i], dp[j] + (i - j) * nums[j])
        return dp[n - 1]