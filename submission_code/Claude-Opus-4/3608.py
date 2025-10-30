class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        from math import gcd
        from functools import lru_cache
        
        MOD = 10**9 + 7
        n = len(nums)
        
        @lru_cache(None)
        def dp(idx, gcd1, gcd2):
            # Base case: processed all elements
            if idx == n:
                # Both subsequences must be non-empty and have equal GCDs
                return 1 if gcd1 > 0 and gcd1 == gcd2 else 0
            
            result = 0
            
            # Option 1: Skip current element
            result = (result + dp(idx + 1, gcd1, gcd2)) % MOD
            
            # Option 2: Add to seq1
            new_gcd1 = gcd(gcd1, nums[idx]) if gcd1 > 0 else nums[idx]
            result = (result + dp(idx + 1, new_gcd1, gcd2)) % MOD
            
            # Option 3: Add to seq2
            new_gcd2 = gcd(gcd2, nums[idx]) if gcd2 > 0 else nums[idx]
            result = (result + dp(idx + 1, gcd1, new_gcd2)) % MOD
            
            return result
        
        return dp(0, 0, 0)