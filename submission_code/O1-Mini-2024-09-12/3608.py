from typing import List
from math import gcd
from collections import defaultdict

class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        max_num = max(nums)
        n = len(nums)
        
        # Count the number of elements divisible by each g
        count_div = [0] * (max_num + 1)
        for num in nums:
            for g in range(1, max_num +1):
                if num % g == 0:
                    count_div[g] +=1
        
        # Compute the number of subsets with GCD exactly g
        f = [0] * (max_num +1)
        for g in range(max_num, 0, -1):
            total = pow(2, count_div[g], MOD) -1
            # Subtract subsets where GCD is a multiple of g
            multiple = 2*g
            while multiple <= max_num:
                total = (total - f[multiple]) % MOD
                multiple += g
            f[g] = total
        
        # Now, for each g, compute the number of pairs of disjoint subsets with GCD=g
        # This can be calculated as the number of ways to choose two disjoint subsets
        # Each element can be in S1, S2, or neither. To ensure S1 and S2 have GCD=g:
        # First, assign elements divisible by g to S1, S2, or neither
        # Then ensure S1 and S2 are non-empty and have GCD=g
        # This is equivalent to f[g] * f[g] because subsets are disjoint
        result = 0
        for g in range(1, max_num +1):
            result = (result + f[g] * f[g]) % MOD
        
        return result