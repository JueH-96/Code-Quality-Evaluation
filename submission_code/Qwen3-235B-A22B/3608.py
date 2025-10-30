import math
from collections import defaultdict
from typing import List

class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        dp = defaultdict(int)
        dp[(0, 0)] = 1  # initial state: no elements taken in either subsequence
        
        for x in nums:
            new_dp = defaultdict(int)
            for (ga, gb), count in dp.items():
                # Option 1: add to neither
                new_dp[(ga, gb)] = (new_dp[(ga, gb)] + count) % MOD
                
                # Option 2: add to A
                new_ga = math.gcd(ga, x)
                new_gb_a = gb
                new_dp[(new_ga, new_gb_a)] = (new_dp[(new_ga, new_gb_a)] + count) % MOD
                
                # Option 3: add to B
                new_ga_b = ga
                new_gb = math.gcd(gb, x)
                new_dp[(new_ga_b, new_gb)] = (new_dp[(new_ga_b, new_gb)] + count) % MOD
            
            dp = new_dp  # update dp to new_dp for the next step
        
        # Sum all states where ga == gb and both are non-zero
        result = 0
        for (ga, gb), count in dp.items():
            if ga == gb and ga != 0:
                result = (result + count) % MOD
        
        return result