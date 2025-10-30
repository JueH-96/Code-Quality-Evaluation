import sys
from math import comb
from typing import List

class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        res = 0
        
        for m in range(n):
            x = nums[m]
            if m < 2 or m > n - 3:
                continue  # Not enough elements before or after
            
            B = nums[:m]
            A = nums[m+1:]
            
            count_x_B = B.count(x)
            count_x_A = A.count(x)
            
            for a in range(0, min(2, count_x_B) + 1):
                for b in range(0, min(2, count_x_A) + 1):
                    total_x = 1 + a + b  # 1 is the middle element itself
                    
                    # The count of x must be greater than any other element's count
                    # To ensure this, the other elements must have counts < total_x
                    # We assume that other elements' counts are at most 1 if total_x >=2
                    # Thus, if total_x > 1, it's valid
                    if total_x > 1:
                        # Calculate ways to choose 'a' x's from B and (2-a) non-x elements
                        ways_B = 0
                        if 2 - a >= 0:
                            ways_B = comb(count_x_B, a) * comb(len(B) - count_x_B, 2 - a)
                        
                        # Calculate ways to choose 'b' x's from A and (2-b) non-x elements
                        ways_A = 0
                        if 2 - b >= 0:
                            ways_A = comb(count_x_A, b) * comb(len(A) - count_x_A, 2 - b)
                        
                        res += ways_B * ways_A
                        res %= MOD
        
        return res % MOD