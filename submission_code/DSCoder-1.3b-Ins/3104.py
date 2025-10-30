class Solution:
    def countWays(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        dp = [0]*n
        dp[0] = 1
        if n > 1:
            dp[1] = 2 if nums[1] != nums[0] else 1
        for i in range(2, n):
            if nums[i] != nums[i-1]:
                dp[i] = dp[i-1]
            else:
                dp[i] = (dp[i-1] + dp[i-2]) % MOD
        return dp[-1]