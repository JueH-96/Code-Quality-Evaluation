class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        nums.sort()
        dp = [[0]*(k+1) for _ in range(n+1)]
        prefix = [0]*(n+1)
        for i in range(n-1, -1, -1):
            prefix[i] = prefix[i+1] + nums[i] if i+1 < n else nums[i]
        for i in range(n-1, -1, -1):
            dp[i][1] = (dp[i+1][1] + nums[i]) % MOD if i+1 < n else nums[i]
            for j in range(2, min(k+1, n-i+1)):
                dp[i][j] = (dp[i+1][j] + nums[i]*pow(2, j-1, MOD) - (prefix[i+j]-prefix[i+1])*pow(2, j-2, MOD)) % MOD
        return dp[0][k]