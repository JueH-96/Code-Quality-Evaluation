class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        dp = [[0]*51 for _ in range(51)]
        dp[0][0] = 1
        for i in range(1, 51):
            dp[i][0] = 1
            for j in range(1, 51):
                dp[i][j] = (dp[i-1][j] + dp[i][j-1]) % MOD
        ans = 1
        for i in range(n-1, -1, -1):
            ans = ans * dp[nums[i]][nums[i]] % MOD
        return ans