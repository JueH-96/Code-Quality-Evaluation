from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        ops = 0
        # Use a greedy approach: for index from 0 to n-3, if current element is 0, flip triple
        for i in range(n - 2):
            if nums[i] == 0:
                # Flip the triple starting at i
                nums[i]   = 1 - nums[i]
                nums[i+1] = 1 - nums[i+1]
                nums[i+2] = 1 - nums[i+2]
                ops += 1
        
        # Check if all elements are 1
        if all(x == 1 for x in nums):
            return ops
        else:
            return -1