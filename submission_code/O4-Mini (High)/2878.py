from typing import List

class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        # expire[i] will hold how many operations expire (stop affecting) at index i
        expire = [0] * n
        dec = 0  # current total decrements affecting this index
        
        for i in range(n):
            # remove effects of operations that have expired at i
            dec -= expire[i]
            
            # effective value at i after all applied decrements so far
            curr = nums[i] - dec
            if curr < 0:
                # too many decrements have hit this position
                return False
            
            if curr > 0:
                # we need curr operations starting at i to bring nums[i] to zero
                # each operation is on a window [i, i+k-1], so we must have room
                if i + k > n:
                    return False
                # apply curr new operations
                dec += curr
                # schedule their expiration at i+k
                end = i + k
                if end < n:
                    expire[end] += curr
        
        return True