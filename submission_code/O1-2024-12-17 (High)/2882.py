class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        
        # Precompute the maximum base we might consider
        # since b^x should not exceed n.
        max_base = 1
        while (max_base + 1) ** x <= n:
            max_base += 1
        
        from functools import lru_cache
        
        @lru_cache(None)
        def ways(remaining: int, base: int) -> int:
            # If we've hit the target sum exactly, count as 1 way.
            if remaining == 0:
                return 1
            # If we've exceeded the largest possible base or can't proceed, no ways.
            if base > max_base:
                return 0
            
            count = 0
            
            # Skip using this base
            count += ways(remaining, base + 1)
            # Use this base (if it doesn't exceed what's left)
            power_val = base ** x
            if power_val <= remaining:
                count += ways(remaining - power_val, base + 1)
            
            return count % MOD
        
        return ways(n, 1) % MOD