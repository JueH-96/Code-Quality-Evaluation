from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # If k is greater than the maximum element in nums, it's impossible
        if k > max(nums):
            return -1
        
        # Sort the array in descending order
        nums.sort(reverse=True)
        
        # Initialize operation count
        operations = 0
        
        # Iterate over the sorted array
        for i in range(len(nums)):
            # If the current number is greater than k, we need to perform an operation
            if nums[i] > k:
                # Set the current number to k
                nums[i] = k
                # Increment the operation count
                operations += 1
        
        # After all operations, check if all elements are equal to k
        if all(num == k for num in nums):
            return operations
        else:
            return -1