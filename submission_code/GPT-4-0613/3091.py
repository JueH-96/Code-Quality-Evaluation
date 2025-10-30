class Solution:
    def countSubMultisets(self, nums, l, r):
        MOD = 10**9 + 7
        dp = [0] * (2 * 10**4 + 1)
        dp[0] = 1
        for num in nums:
            for i in range(2 * 10**4, num - 1, -1):
                dp[i] = (dp[i] + dp[i - num]) % MOD
        return sum(dp[l:r+1]) % MOD