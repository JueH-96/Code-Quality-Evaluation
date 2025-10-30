from typing import List
from collections import defaultdict
import math

class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        max_num = max(nums) if nums else 0
        total = 0
        
        # Iterate all possible g from 1 to 200
        for g in range(1, 201):
            S_g = [num for num in nums if num % g == 0]
            k = len(S_g)
            if k < 2:
                continue
            # Create transformed array B
            B = [num // g for num in S_g]
            # Compute DP for B
            dp = defaultdict(int)
            dp[(0, 0)] = 1
            for x in B:
                new_dp = defaultdict(int)
                for (g1, g2), cnt in dp.items():
                    # Option: add to neither
                    new_dp[(g1, g2)] = (new_dp[(g1, g2)] + cnt) % MOD
                    # Option: add to s1
                    if g1 == 0:
                        new_g1 = x
                    else:
                        new_g1 = math.gcd(g1, x)
                    new_state_a = (new_g1, g2)
                    new_dp[new_state_a] = (new_dp[new_state_a] + cnt) % MOD
                    # Option: add to s2
                    if g2 == 0:
                        new_g2 = x
                    else:
                        new_g2 = math.gcd(g2, x)
                    new_state_b = (g1, new_g2)
                    new_dp[new_state_b] = (new_dp[new_state_b] + cnt) % MOD
                dp = new_dp
            # Sum all valid (g1, g2) where both are non-zero and equal to 1
            res = 0
            for (g1, g2), cnt in dp.items():
                if g1 != 0 and g2 != 0 and g1 == 1 and g2 == 1:
                    res = (res + cnt) % MOD
            total = (total + res) % MOD
        
        return total