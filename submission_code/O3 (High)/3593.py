from typing import List
import math

class Solution:
    def maxScore(self, nums: List[int]) -> int:
        """
        For every possible choice of “element to remove” (including the choice of
        removing nothing) we compute the gcd and the lcm of the remaining
        elements and keep the largest product gcd * lcm.
        
        With n ≤ 100 a straightforward O(n²) solution is more than fast enough,
        so we simply iterate through the array for each of the at most n + 1
        removal-options.
        """
        def lcm(a: int, b: int) -> int:
            return a // math.gcd(a, b) * b
        
        n = len(nums)
        best = 0
        
        # remove == -1 means “keep every element”
        for remove in range(-1, n):
            gcd_val = 0      # will be set to first kept element
            lcm_val = 0
            for i, x in enumerate(nums):
                if i == remove:          # skip the chosen element
                    continue
                if gcd_val == 0:         # first kept element
                    gcd_val = lcm_val = x
                else:
                    gcd_val = math.gcd(gcd_val, x)
                    lcm_val = lcm(lcm_val, x)
            
            if gcd_val == 0:             # array became empty
                continue
            best = max(best, gcd_val * lcm_val)
        
        return best