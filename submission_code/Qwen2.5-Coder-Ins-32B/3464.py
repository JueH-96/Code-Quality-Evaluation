class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        
        max_cost = nums[0]
        current_cost = nums[0]
        
        for i in range(1, len(nums)):
            # Calculate the cost of starting a new subarray at index i
            start_new_subarray = nums[i]
            # Calculate the cost of continuing the current subarray
            continue_subarray = current_cost + nums[i] * (-1 if (i % 2 == 1) else 1)
            
            # Choose the better option
            current_cost = max(start_new_subarray, continue_subarray)
            
            # Update the maximum cost found so far
            max_cost = max(max_cost, current_cost)
        
        return max_cost