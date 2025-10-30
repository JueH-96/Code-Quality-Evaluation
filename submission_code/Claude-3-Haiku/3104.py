class Solution:
    def countWays(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        
        # Base case: If no students are selected, it's always a valid way
        for i in range(n + 1):
            dp[i][0] = 1
        
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                # If the current student is selected
                if j > nums[i - 1]:
                    dp[i][j] += dp[i - 1][j - 1]
                # If the current student is not selected
                if j < nums[i - 1]:
                    dp[i][j] += dp[i - 1][j]
        
        return dp[n][n]