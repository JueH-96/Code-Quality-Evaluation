from typing import List

class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        nums.sort()
        
        total = 0
        # f = sum of mins over all non-empty subsets of processed prefix
        f = 0
        
        for x in nums:
            # For current x as the maximum:
            #   S = f is sum of mins over all non-empty subsets of previous elements
            #   empty subset gives min = x
            # Contribution = x^2 * (x + S)
            contrib = x * x % MOD * (x + f) % MOD
            total = (total + contrib) % MOD
            
            # Update f for inclusion of x into the prefix:
            # f_new = 2 * f_old + x
            f = (2 * f + x) % MOD
        
        return total