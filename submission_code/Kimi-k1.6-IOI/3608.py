import math
from collections import defaultdict
from typing import List

class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        dp = defaultdict(int)
        dp[(0, 0)] = 1  # Initial state: both subsequences are empty
        
        for x in nums:
            new_dp = defaultdict(int)
            for (g1, g2), cnt in dp.items():
                # Option 1: Add current element to the first subsequence
                new_g1 = x if g1 == 0 else math.gcd(g1, x)
                new_dp[(new_g1, g2)] = (new_dp[(new_g1, g2)] + cnt) % MOD
                
                # Option 2: Add current element to the second subsequence
                new_g2 = x if g2 == 0 else math.gcd(g2, x)
                new_dp[(g1, new_g2)] = (new_dp[(g1, new_g2)] + cnt) % MOD
                
                # Option 3: Do not add current element to either subsequence
                new_dp[(g1, g2)] = (new_dp[(g1, g2)] + cnt) % MOD
            dp = new_dp
        
        total = 0
        for (g1, g2), cnt in dp.items():
            if g1 != 0 and g2 != 0 and g1 == g2:
                total = (total + cnt) % MOD
        return total