from typing import List

class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        """
        We compute the difference array diff[i] = target[i] - nums[i].
        Each operation can either increment or decrement a contiguous subarray by 1.
        We can count the needed positive (increment) and negative (decrement) range-ops
        separately.  If we let
            pos[i] = max(diff[i], 0)
            neg[i] = max(-diff[i], 0)
        then the minimum number of increments is
            sum_{i=0..n-1} max(pos[i] - pos[i-1], 0)
        (with pos[-1] = 0),
        and similarly for decrements with neg[].
        """
        n = len(nums)
        prev_pos = 0
        prev_neg = 0
        inc_ops = 0
        dec_ops = 0
        
        for i in range(n):
            d = target[i] - nums[i]
            p = d if d > 0 else 0
            ng = -d if d < 0 else 0
            
            # count new increment operations if pos rises
            if p > prev_pos:
                inc_ops += (p - prev_pos)
            # count new decrement operations if neg rises
            if ng > prev_neg:
                dec_ops += (ng - prev_neg)
            
            prev_pos = p
            prev_neg = ng
        
        return inc_ops + dec_ops