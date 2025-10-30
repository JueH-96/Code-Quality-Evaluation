from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        operations = 0
        
        for i in range(n - 2):
            if nums[i] == 0:
                # Flip the current and next two elements
                nums[i] = 1 - nums[i]
                nums[i + 1] = 1 - nums[i + 1]
                nums[i + 2] = 1 - nums[i + 2]
                operations += 1
        
        # Check if the last two elements are 1
        if nums[n - 2] != 1 or nums[n - 1] != 1:
            return -1
        
        return operations