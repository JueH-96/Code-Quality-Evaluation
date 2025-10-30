class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 0
        
        # Calculate the cost of the entire array without splitting
        total_cost = 0
        for i in range(n):
            if i % 2 == 0:
                total_cost += nums[i]
            else:
                total_cost -= nums[i]
        
        # Dynamic programming array to store the maximum cost up to each index
        dp = [0] * n
        dp[0] = nums[0]
        
        # Calculate the maximum cost using dp
        for i in range(1, n):
            dp[i] = dp[i-1] + (nums[i] if i % 2 == 0 else -nums[i])
            # Check if splitting at each index increases the total cost
            for j in range(i):
                current_cost = dp[j] + (nums[i] if (i-j) % 2 == 0 else -nums[i])
                if current_cost > dp[i]:
                    dp[i] = current_cost
        
        return dp[n-1]