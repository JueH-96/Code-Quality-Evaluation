class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # dp[i][j] = max length ending at index i with exactly j transitions
        dp = [[0] * (k + 1) for _ in range(n)]
        
        # Base case: each element by itself
        for i in range(n):
            dp[i][0] = 1
        
        for i in range(n):
            for j in range(k + 1):
                for prev in range(i):
                    if nums[prev] == nums[i]:
                        # No transition needed
                        dp[i][j] = max(dp[i][j], dp[prev][j] + 1)
                    elif j > 0:
                        # One transition needed
                        dp[i][j] = max(dp[i][j], dp[prev][j-1] + 1)
        
        # Return maximum value in dp table
        result = 0
        for i in range(n):
            for j in range(k + 1):
                result = max(result, dp[i][j])
        
        return result