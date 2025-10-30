class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = [[0] * (k + 1) for _ in range(n)]
        
        for i in range(n):
            dp[i][0] = 1
            for j in range(1, min(i + 1, k + 1)):
                dp[i][j] = dp[i - 1][j]
                if i > 0 and nums[i] != nums[i - 1]:
                    dp[i][j] = max(dp[i][j], dp[i - 1][j - 1] + 1)
        
        return max(max(row) for row in dp)