class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        dp = [0] * (n + 2)
        max_sum = [0] * (n + 2)
        for i in range(n - 1, -1, -1):
            dp[i] = max(dp[i + 1], nums[i] + max_sum[i + 2])
            max_sum[i] = max(max_sum[i + 1], dp[i])
        ans = 0
        for pos, x in queries:
            nums[pos] = x
            dp[pos] = max(dp[pos + 1], x + max_sum[pos + 2])
            max_sum[pos] = max(max_sum[pos + 1], dp[pos])
            ans = (ans + dp[pos]) % MOD
        return ans