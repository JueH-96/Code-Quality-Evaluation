class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        min_cost = float('inf')
        
        # Iterate over all possible ways to split the array into 3 subarrays
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Calculate the cost of the current split
                cost = nums[0] + nums[i] + nums[j]
                # Update the minimum cost if the current cost is lower
                min_cost = min(min_cost, cost)
        
        return min_cost