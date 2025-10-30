class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        dp = [0, 0]  # dp[0] for even, dp[1] for odd
        dp[nums[0] % 2] = nums[0]
        
        for num in nums[1:]:
            dp[num % 2] = max(dp[num % 2], dp[1 - num % 2] - x) + num
        
        return max(dp)