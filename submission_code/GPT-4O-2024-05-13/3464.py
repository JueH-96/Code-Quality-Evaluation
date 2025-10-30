class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return nums[0]
        
        # Calculate the cost of the entire array without any splits
        total_cost = 0
        for i in range(n):
            total_cost += nums[i] * (-1) ** (i % 2)
        
        # We will use dynamic programming to find the maximum total cost
        dp = [0] * n
        dp[0] = nums[0]
        
        for i in range(1, n):
            dp[i] = max(dp[i-1] + nums[i] * (-1) ** (i % 2), nums[i] * (-1) ** (i % 2))
        
        return max(dp)