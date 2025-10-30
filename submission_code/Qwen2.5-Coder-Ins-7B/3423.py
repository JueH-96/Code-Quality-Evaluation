class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        dp = [0] * (n + 1)
        dp[1] = nums[0]
        
        for i in range(2, n + 1):
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i - 1])
        
        total_sum = 0
        for pos, x in queries:
            nums[pos] = x
            if pos == 0:
                dp[1] = max(dp[1], nums[0])
            elif pos == n - 1:
                dp[n] = max(dp[n], nums[n - 1])
            else:
                dp[pos + 1] = max(dp[pos], dp[pos + 1], dp[pos - 1] + nums[pos], dp[pos + 2] + nums[pos])
            total_sum = (total_sum + dp[-1]) % MOD
        
        return total_sum