class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        # Initialize a list to store the minimum operations for each index
        dp = [float('inf')] * (max(max(nums), max(target)) + 1)
        dp[0] = 0
        
        # Iterate through the nums and target lists
        for i in range(len(nums)):
            # Iterate through the dp list in reverse order
            for j in range(len(dp) - 1, nums[i] - 1, -1):
                # Update the minimum operations for each index
                dp[j] = min(dp[j], dp[j - nums[i]] + 1)
        
        # Initialize the result to the minimum operations for the target
        result = dp[target[0]]
        
        # Iterate through the target list
        for i in range(1, len(target)):
            # Update the result to the minimum operations for the target
            result = min(result, dp[target[i]])
        
        # If the result is infinity, return -1
        if result == float('inf'):
            return -1
        
        # Return the result
        return result