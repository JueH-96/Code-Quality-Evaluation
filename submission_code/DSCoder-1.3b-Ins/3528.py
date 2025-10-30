class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [[0]*n for _ in range(n)]
        for i in range(n-1, -1, -1):
            for j in range(i, n):
                dp[i][j] = max(nums[i]*nums[j] + dp[i+1][j], nums[j]*nums[i] + dp[i][j-1])
        return dp[0][n-1]