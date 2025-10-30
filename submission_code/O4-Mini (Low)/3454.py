from typing import List

class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        """
        We transform nums to target by range +1 or -1 operations.
        Let d[i] = target[i] - nums[i].
        Split into increments d+ = max(d,0) and decrements d- = max(-d,0).
        It's known that the minimum number of +1-range ops to achieve d+ is
          sum over i of max(d+[i] - d+[i-1], 0),
        with d+[-1] = 0.  Similarly for d-.
        Total ops = ops_inc + ops_dec.
        """
        n = len(nums)
        total_ops = 0
        prev_inc = 0
        prev_dec = 0
        for i in range(n):
            diff = target[i] - nums[i]
            # positive part, for increment operations
            inc = diff if diff > 0 else 0
            # positive part of -diff, for decrement operations
            dec = -diff if diff < 0 else 0
            
            # if inc increased compared to prev_inc, we need extra inc ops
            if inc > prev_inc:
                total_ops += inc - prev_inc
            # similarly for dec vs prev_dec
            if dec > prev_dec:
                total_ops += dec - prev_dec
            
            prev_inc = inc
            prev_dec = dec
        
        return total_ops