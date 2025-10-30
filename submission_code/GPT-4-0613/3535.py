class Solution:
    def countOfPairs(self, nums):
        mod = 10**9 + 7
        n = len(nums)
        dp = [[0]*51 for _ in range(n+1)]
        dp[0][0] = 1
        for i in range(1, n+1):
            for j in range(51):
                for k in range(j+1):
                    if k > nums[i-1]:
                        break
                    dp[i][j] = (dp[i][j] + dp[i-1][j-k]) % mod
        res = 0
        for i in range(51):
            res = (res + dp[n][i]) % mod
        return res