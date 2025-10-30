class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        n = len(nums)
        dp = [-float('inf')] * 2
        dp[nums[0] % 2] = nums[0]
        
        for i in range(1, n):
            parity = nums[i] % 2
            dp[parity] = max(dp[parity] + nums[i], dp[1 - parity] + nums[i] - x)
        
        return max(dp[0], dp[1])