class Solution:
    def minimumValueSum(self, nums, andValues):
        n, m = len(nums), len(andValues)
        dp = [[float('inf')] * (m + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        prefix_and = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix_and[i] = prefix_and[i - 1] & nums[i - 1]
            for j in range(1, min(i, m) + 1):
                k = i
                while k > j - 1:
                    if prefix_and[i] == andValues[j - 1]:
                        dp[i][j] = min(dp[i][j], dp[k - 1][j - 1] + nums[i - 1])
                    if k > 1:
                        prefix_and[i] = prefix_and[i] & nums[k - 2]
                    k -= 1
        return dp[n][m] if dp[n][m] != float('inf') else -1