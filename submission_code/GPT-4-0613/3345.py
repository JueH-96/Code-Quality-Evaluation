class Solution:
    def sumOfPower(self, nums, k):
        MOD = 10**9 + 7
        dp = [0] * (k + 1)
        dp[0] = 1
        for num in nums:
            for i in range(k, num - 1, -1):
                dp[i] = (dp[i] + dp[i - num]) % MOD
        return (pow(2, len(nums), MOD) - 1 - dp[k] + MOD) % MOD