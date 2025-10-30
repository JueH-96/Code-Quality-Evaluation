class Solution:
    def maximumStrength(self, nums, k):
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            for j in range(1, min(i, k) + 1):
                dp[i][j] = dp[i - 1][j]
                l, r = 0, i - j
                while l <= r:
                    mid = (l + r) // 2
                    if (dp[mid][j - 1] + (prefix_sum[i] - prefix_sum[mid]) * j) > dp[i][j]:
                        dp[i][j] = dp[mid][j - 1] + (prefix_sum[i] - prefix_sum[mid]) * j
                        r = mid - 1
                    else:
                        l = mid + 1
        return dp[n][k]