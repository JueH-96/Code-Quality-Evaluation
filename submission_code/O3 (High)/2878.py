from typing import List

class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        # difference array to cancel the effect of operations that end at a position
        diff = [0]*(n+1)
        current_delta = 0          # total decrement currently affecting position i
        
        for i in range(n):
            current_delta += diff[i]        # remove effects that ended before i
            val = nums[i] + current_delta   # value after all previous operations
            
            if val < 0:                     # over-decremented – impossible
                return False
            
            if val > 0:                     # need to apply `val` operations starting here
                if i + k > n:               # subarray would exceed bounds
                    return False
                current_delta -= val        # apply the decrement for the next k elements
                diff[i + k] += val          # cancel the effect after position i+k-1
        
        # if we never encountered an impossibility, array can be made all zeros
        return True