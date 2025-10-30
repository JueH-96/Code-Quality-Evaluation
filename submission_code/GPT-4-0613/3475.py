class Solution:
    def minOperations(self, nums):
        n = len(nums)
        dp = [[0, float('inf')] for _ in range(n+1)]
        for i in range(1, n+1):
            if nums[i-1] == 1:
                dp[i][0] = min(dp[i-1])
                dp[i][1] = min(dp[i-1]) + 1
            else:
                if i >= 3:
                    dp[i][0] = min(dp[i-3]) + 1
                dp[i][1] = min(dp[i-1]) + 1
        return min(dp[-1]) if min(dp[-1]) != float('inf') else -1