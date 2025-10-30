class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        
        # Precompute all possible base numbers whose x-th power <= n
        powers = []
        base = 1
        while True:
            val = base ** x
            if val > n:
                break
            powers.append(val)
            base += 1
        
        # Memoization cache: dp(idx, remaining) -> number of ways
        from functools import lru_cache
        
        @lru_cache(None)
        def ways(idx, remaining):
            # If we've hit the sum exactly, that's one valid way
            if remaining == 0:
                return 1
            # If we've run out of powers or exceeded sum, no valid way
            if idx == len(powers) or remaining < 0:
                return 0
            
            # Option 1: Skip current power
            ans = ways(idx + 1, remaining)
            # Option 2: Use current power (if it doesn't exceed 'remaining')
            if powers[idx] <= remaining:
                ans += ways(idx + 1, remaining - powers[idx])
            
            return ans % MOD
        
        return ways(0, n) % MOD