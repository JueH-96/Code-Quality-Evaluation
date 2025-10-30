class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Calculate the cost of the entire array as a single subarray
        total_cost = sum(nums[i] if i % 2 == 0 else -nums[i] for i in range(n))
        
        # Initialize the maximum cost as the total cost of the whole array
        max_cost = total_cost
        
        # We will use a dynamic programming approach to find the maximum cost
        # dp[i] will store the maximum cost we can achieve by considering nums[0..i]
        dp = [0] * n
        
        # Calculate the cost for the first element
        dp[0] = nums[0]
        
        for i in range(1, n):
            # Calculate the cost of the subarray nums[0..i]
            dp[i] = dp[i - 1] + (nums[i] if i % 2 == 0 else -nums[i])
            # Update the maximum cost
            max_cost = max(max_cost, dp[i])
        
        return max_cost