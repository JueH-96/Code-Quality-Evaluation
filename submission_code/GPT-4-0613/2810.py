class Solution:
    def minCost(self, nums, x):
        n = len(nums)
        nums = sorted([(c, i) for i, c in enumerate(nums)])
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        for i in range(n):
            ndp = dp[:]
            for j in range(i + 1):
                ndp[j + 1] = min(ndp[j + 1], dp[j] + nums[i][0] + x * max(0, i - j))
            dp = ndp
        return min(dp[i] + x * max(0, n - i) for i in range(n + 1))