from typing import List

class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        # First, sort the array in non-decreasing order
        nums.sort()
        
        # Calculate the index of the median
        median_index = len(nums) // 2
        
        # If the length of the array is even, the median is the larger of the two middle values
        if len(nums) % 2 == 0:
            median_index = median_index - 1 if nums[median_index - 1] > k else median_index
        
        # Calculate the difference between the current median and the target median
        operations = abs(nums[median_index] - k)
        
        # If the current median is already equal to the target median, return 0
        if operations == 0:
            return 0
        
        # If the current median is less than the target median, we need to increase it
        if nums[median_index] < k:
            # Increase the elements to the left of the median to the target median
            for i in range(median_index - 1, -1, -1):
                if nums[i] < k:
                    operations += k - nums[i]
                else:
                    break
            
            # Decrease the elements to the right of the median to the target median
            for i in range(median_index + 1, len(nums)):
                if nums[i] > k:
                    operations += nums[i] - k
                else:
                    break
        
        # If the current median is greater than the target median, we need to decrease it
        else:
            # Decrease the elements to the right of the median to the target median
            for i in range(median_index + 1, len(nums)):
                if nums[i] > k:
                    operations += nums[i] - k
                else:
                    break
            
            # Increase the elements to the left of the median to the target median
            for i in range(median_index - 1, -1, -1):
                if nums[i] < k:
                    operations += k - nums[i]
                else:
                    break
        
        return operations