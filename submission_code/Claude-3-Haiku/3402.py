class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        n = len(nums)
        nums.sort()
        
        # Calculate the total cost to make all elements equal to the minimum value
        min_cost = 0
        for i in range(1, n):
            min_cost += (nums[i] - nums[0]) * cost1
        
        # Calculate the minimum cost to make all elements equal
        min_cost_equal = min_cost
        
        # Try to make all elements equal using the two-element operation
        for i in range(1, n):
            # Calculate the cost to make all elements equal to the average of the first i elements
            cost = 0
            for j in range(i):
                cost += (nums[i] - nums[j]) * cost2
            min_cost_equal = min(min_cost_equal, min_cost - cost)
        
        return min_cost_equal % (10**9 + 7)