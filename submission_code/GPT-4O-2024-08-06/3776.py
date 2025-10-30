class Solution:
    def minCost(self, nums: List[int]) -> int:
        total_cost = 0
        
        while len(nums) > 2:
            # Consider the first three elements
            first_three = nums[:3]
            # Sort them to easily find the two smallest
            first_three.sort()
            # Remove the two smallest elements
            nums.remove(first_three[0])
            nums.remove(first_three[1])
            # Add the cost of the operation, which is the maximum of the two removed
            total_cost += first_three[1]
        
        # If there are 1 or 2 elements left, remove them with the cost of the max
        if nums:
            total_cost += max(nums)
        
        return total_cost