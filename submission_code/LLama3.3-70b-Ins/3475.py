from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        operations = 0
        i = 0
        
        # Traverse the array
        while i < n:
            # If the current element is 0
            if nums[i] == 0:
                # If there are less than 3 elements left, it's impossible to make all elements 1
                if n - i < 3:
                    return -1
                # Flip the current element and the next two elements
                nums[i] = 1 - nums[i]
                nums[i+1] = 1 - nums[i+1]
                nums[i+2] = 1 - nums[i+2]
                # Increment the operations count
                operations += 1
                # Move to the next element
                i += 1
            # If the current element is 1, move to the next element
            else:
                i += 1
        
        return operations