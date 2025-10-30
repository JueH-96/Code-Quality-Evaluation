from typing import List

class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        # Sort the array in descending order
        nums.sort(reverse=True)
        
        # Iterate over the array
        for i in range(len(nums) - 2):
            # Check if the current element is smaller than the sum of the next two elements
            if nums[i] < nums[i+1] + nums[i+2]:
                # If it is, return the sum of the current element and the next two elements
                return nums[i] + nums[i+1] + nums[i+2]
        
        # If no such combination is found, return -1
        return -1