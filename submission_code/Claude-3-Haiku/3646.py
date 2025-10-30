class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        MOD = 10 ** 9 + 7
        n = len(nums)
        dp = [[0] * 101 for _ in range(n)]
        
        for i in range(n):
            for j in range(101):
                if j == nums[i]:
                    dp[i][j] = 1
                elif i > 0 and abs(j - nums[i]) == 1:
                    for k in range(101):
                        dp[i][j] = (dp[i][j] + dp[i-1][k]) % MOD
        
        total = 0
        for j in range(101):
            total = (total + sum(dp[n-1][j])) % MOD
        
        return total