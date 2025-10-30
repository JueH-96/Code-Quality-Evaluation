from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        ops = 0
        
        # Greedily flip any zero at position i by flipping [i, i+1, i+2]
        for i in range(n - 2):
            if nums[i] == 0:
                ops += 1
                nums[i] ^= 1
                nums[i+1] ^= 1
                nums[i+2] ^= 1
        
        # If the last two positions are 1, we've succeeded
        if nums[n-2] == 1 and nums[n-1] == 1:
            return ops
        return -1