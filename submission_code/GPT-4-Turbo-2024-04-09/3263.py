class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 3:
            return nums[0] + nums[1] + nums[2]
        
        # Initialize the minimum cost to a large number
        min_cost = float('inf')
        
        # We need to pick two indices i and j where 0 < i < j < n
        # such that we split the array into three parts:
        # part1: nums[0:i], part2: nums[i:j], part3: nums[j:n]
        for i in range(1, n-1):
            for j in range(i+1, n):
                # Calculate the cost of the current division
                cost = nums[0] + nums[i] + nums[j]
                # Update the minimum cost found so far
                min_cost = min(min_cost, cost)
        
        return min_cost