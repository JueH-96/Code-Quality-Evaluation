class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            for j in range(k + 1):
                if j == 0:
                    dp[i][j] = max(dp[i - 1][j], 1)
                else:
                    if nums[i - 1] == nums[i - 2]:
                        dp[i][j] = dp[i - 1][j]
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - 1] + 1)
        
        return dp[n][k]