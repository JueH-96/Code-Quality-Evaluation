class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        nums.sort()
        n = len(nums)
        dp = [0]*(n+1)
        for i in range(n-1, -1, -1):
            dp[i] = max(dp[i+1], nums[i]**2 + (dp[i+2] if i+2 < n else 0))
        return dp[0] % MOD