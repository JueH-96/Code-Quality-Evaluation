from typing import List

class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        # diff[i] will track how many operations "end" at index i
        diff = [0] * (n + 1)
        curr_subtractions = 0
        
        for i in range(n):
            # Remove the effect of operations that ended at i
            curr_subtractions -= diff[i]
            
            # If current value after subtractions is still > 0, we need more operations
            remaining = nums[i] - curr_subtractions
            if remaining > 0:
                # We must apply `remaining` operations on the subarray starting at i
                # Check if we have room for a subarray of length k
                if i + k > n:
                    return False
                curr_subtractions += remaining
                # Schedule these operations to stop affecting indices >= i+k
                diff[i + k] += remaining
        
        return True