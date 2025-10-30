import math
from typing import List
from collections import defaultdict

class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        if not nums:
            return 0
        max_num = max(nums)
        total = 0
        for g in range(1, max_num + 1):
            # Collect elements divisible by g
            S = [x for x in nums if x % g == 0]
            if len(S) < 2:
                continue  # Need at least two elements to form two non-empty subsets
            T = [x // g for x in S]
            # Compute DP for T
            dp = defaultdict(int)
            dp[(0, 0)] = 1
            for x in T:
                new_dp = defaultdict(int)
                for (gA, gB), cnt in dp.items():
                    # Option 1: add x to A
                    new_gA = math.gcd(gA, x)
                    new_dp[(new_gA, gB)] = (new_dp[(new_gA, gB)] + cnt) % MOD
                    # Option 2: add x to B
                    new_gB = math.gcd(gB, x)
                    new_dp[(gA, new_gB)] = (new_dp[(gA, new_gB)] + cnt) % MOD
                    # Option 3: do not add x to either
                    new_dp[(gA, gB)] = (new_dp[(gA, gB)] + cnt) % MOD
                dp = new_dp
            total = (total + dp.get((1, 1), 0)) % MOD
        return total