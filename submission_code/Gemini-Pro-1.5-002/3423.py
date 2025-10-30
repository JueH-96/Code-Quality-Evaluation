class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        mod = 10**9 + 7
        ans = 0

        for pos, x in queries:
            nums[pos] = x
            dp = [0] * (n + 1)
            for i in range(n):
                dp[i+1] = max(dp[i], dp[i-1] + nums[i] if i > 0 else nums[i])
            ans = (ans + dp[n]) % mod
        
        return ans