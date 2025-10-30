class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        MOD = 10**9 + 7
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            dp[i][1] = dp[i - 1][1] + nums[i - 1]
            for j in range(2, k + 1):
                dp[i][j] = dp[i - 1][j] + dp[i - 1][j - 1] * (nums[i - 1] - nums[i - j])
        return dp[n][k] % MOD