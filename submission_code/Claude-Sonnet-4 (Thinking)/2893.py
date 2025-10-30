class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        # dp[0] = max score with last visited number being even
        # dp[1] = max score with last visited number being odd
        if nums[0] % 2 == 0:
            dp = [nums[0], float('-inf')]
        else:
            dp = [float('-inf'), nums[0]]
        
        for i in range(1, len(nums)):
            if nums[i] % 2 == 0:  # nums[i] is even
                dp[0] = max(dp[0], dp[0] + nums[i], dp[1] + nums[i] - x)
            else:  # nums[i] is odd
                dp[1] = max(dp[1], dp[1] + nums[i], dp[0] + nums[i] - x)
        
        return max(dp)