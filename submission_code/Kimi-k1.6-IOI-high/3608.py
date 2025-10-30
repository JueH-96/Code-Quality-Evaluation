from collections import defaultdict
import math

class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        dp = defaultdict(int)
        dp[(0, 0, 0)] = 1  # (gA, gB, mask) -> count
        
        for num in nums:
            next_dp = defaultdict(int)
            # Option 3: Add to neither (copy current dp to next_dp)
            for key, cnt in dp.items():
                next_dp[key] = (next_dp[key] + cnt) % MOD
            
            # Option 1: Add to A
            for (gA, gB, mask), cnt in dp.items():
                new_gA = math.gcd(gA, num) if gA != 0 else num
                new_mask = mask | 1
                next_dp[(new_gA, gB, new_mask)] = (next_dp[(new_gA, gB, new_mask)] + cnt) % MOD
            
            # Option 2: Add to B
            for (gA, gB, mask), cnt in dp.items():
                new_gB = math.gcd(gB, num) if gB != 0 else num
                new_mask = mask | 2
                next_dp[(gA, new_gB, new_mask)] = (next_dp[(gA, new_gB, new_mask)] + cnt) % MOD
            
            dp = next_dp
        
        total = 0
        for (gA, gB, mask), cnt in dp.items():
            if mask == 3 and gA == gB:  # mask 11 (binary) means both are non-empty
                total = (total + cnt) % MOD
        
        return total