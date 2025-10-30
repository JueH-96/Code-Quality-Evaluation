from math import gcd

class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        
        # dp[gcd1][gcd2] = number of ways to have seq1 with GCD gcd1 and seq2 with GCD gcd2
        # We use 0 to represent empty sequence
        dp = {(0, 0): 1}
        
        for num in nums:
            new_dp = {}
            for (gcd1, gcd2), count in dp.items():
                # Don't include num in either sequence
                key = (gcd1, gcd2)
                new_dp[key] = (new_dp.get(key, 0) + count) % MOD
                
                # Include num in seq1
                new_gcd1 = num if gcd1 == 0 else gcd(gcd1, num)
                key = (new_gcd1, gcd2)
                new_dp[key] = (new_dp.get(key, 0) + count) % MOD
                
                # Include num in seq2
                new_gcd2 = num if gcd2 == 0 else gcd(gcd2, num)
                key = (gcd1, new_gcd2)
                new_dp[key] = (new_dp.get(key, 0) + count) % MOD
            
            dp = new_dp
        
        result = 0
        for (gcd1, gcd2), count in dp.items():
            if gcd1 > 0 and gcd2 > 0 and gcd1 == gcd2:
                result = (result + count) % MOD
        
        return result