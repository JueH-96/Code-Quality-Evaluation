class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        # Initialize the first two costs as the first two elements of the array
        cost1, cost2 = nums[0], nums[1]
        
        # Iterate through the array starting from the third element
        for i in range(2, len(nums)):
            # Calculate the minimum cost of dividing the array into three subarrays
            # by considering the current element as the start of the third subarray
            # and updating the costs of the first two subarrays accordingly
            prev_cost1, prev_cost2 = cost1, cost2
            cost1, cost2 = min(cost1, prev_cost2) + nums[i], prev_cost1
        
        # Return the minimum cost of dividing the array into three subarrays
        return min(cost1, cost2)