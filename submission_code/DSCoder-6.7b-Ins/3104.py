class Solution:
    def countWays(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        fact = [0] * (n + 1)
        factinv = [0] * (n + 1)
        fact[0] = factinv[0] = 1
        for i in range(1, n + 1):
            fact[i] = (fact[i - 1] * i) % MOD
            factinv[i] = factinv[i - 1] * pow(i, MOD - 2, MOD)
        for i in range(2, n + 1):
            fact[i] = (fact[i] * factinv[i - 2]) % MOD
        nums.sort()
        dp = [0] * (n + 1)
        dp[0] = 1
        for i in range(1, n + 1):
            dp[i] = dp[i - 1]
            if nums[i - 1] >= i - 1:
                dp[i] = (dp[i] + fact[i - 1] * factinv[i - 1] * factinv[n - i]) % MOD
            if i - 1 > nums[i - 1]:
                dp[i] = (dp[i] + fact[i - 1] * factinv[i - 1] * factinv[n - i] * fact[nums[i - 1]] * factinv[nums[i - 1] - 1] * factinv[i - nums[i - 1] - 1]) % MOD
        return dp[n]