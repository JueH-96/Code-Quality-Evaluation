class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        from math import gcd
        
        MOD = 10**9 + 7
        
        # dp[(g1, g2)] = number of ways to form seq1 with GCD g1 and seq2 with GCD g2
        # We use 0 to denote empty subsequence
        dp = {(0, 0): 1}
        
        for num in nums:
            new_dp = {}
            
            for (g1, g2), count in dp.items():
                # Option 1: Don't include current number in either subsequence
                key = (g1, g2)
                new_dp[key] = (new_dp.get(key, 0) + count) % MOD
                
                # Option 2: Include in seq1
                new_g1 = gcd(g1, num) if g1 > 0 else num
                key = (new_g1, g2)
                new_dp[key] = (new_dp.get(key, 0) + count) % MOD
                
                # Option 3: Include in seq2
                new_g2 = gcd(g2, num) if g2 > 0 else num
                key = (g1, new_g2)
                new_dp[key] = (new_dp.get(key, 0) + count) % MOD
            
            dp = new_dp
        
        # Count pairs where g1 == g2 and both are non-empty
        result = 0
        for (g1, g2), count in dp.items():
            if g1 == g2 and g1 > 0:
                result = (result + count) % MOD
        
        return result