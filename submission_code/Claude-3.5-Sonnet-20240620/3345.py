class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        dp = [0] * (k + 1)
        dp[0] = 1

        for num in nums:
            for target in range(k, num - 1, -1):
                dp[target] = (dp[target] + dp[target - num]) % MOD

        result = 0
        for i in range(n):
            curr = 0
            for target in range(k + 1):
                curr = (curr + dp[target] * pow(2, n - i - 1, MOD)) % MOD
            result = (result + curr * nums[i]) % MOD
            
            for target in range(k, nums[i] - 1, -1):
                dp[target] = (dp[target] - dp[target - nums[i]]) % MOD

        return result