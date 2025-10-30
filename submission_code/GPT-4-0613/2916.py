class Solution:
    def canSplitArray(self, nums: list[int], m: int) -> bool:
        nums.sort(reverse=True)
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        if prefix_sum[n] < m:
            return False
        dp = [0] + [float('inf')] * n
        for i in range(1, n + 1):
            for j in range(i):
                if prefix_sum[i] - prefix_sum[j] >= m:
                    dp[i] = min(dp[i], dp[j] + 1)
        return dp[n] <= n