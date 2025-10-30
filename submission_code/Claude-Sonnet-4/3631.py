class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        MOD = 10**9 + 7
        n = len(s)
        
        # First, find which bit counts are k-reducible
        # A bit count is k-reducible if it can reach 1 in at most k steps
        max_bits = n  # Maximum possible bit count is n
        reducible = [False] * (max_bits + 1)
        reducible[1] = True  # 1 is 0-reducible (already 1)
        
        # For each possible bit count, check if it's k-reducible
        for bits in range(2, max_bits + 1):
            # Count set bits in 'bits'
            next_bits = bin(bits).count('1')
            # Check if we can reach 1 in at most k-1 more steps
            for steps in range(k):
                if reducible[next_bits]:
                    reducible[bits] = True
                    break
                if next_bits == 1:
                    break
                next_bits = bin(next_bits).count('1')
        
        # Now count numbers less than s with k-reducible bit counts
        # Use digit DP: dp[pos][bits][tight][started]
        # pos: current position, bits: current bit count
        # tight: whether we're still bounded by s, started: whether we've placed a 1
        
        from functools import lru_cache
        
        @lru_cache(None)
        def dp(pos, bits, tight, started):
            if pos == n:
                # If we haven't started (all zeros), return 0
                # Otherwise, check if bit count is k-reducible
                if not started:
                    return 0
                return 1 if reducible[bits] else 0
            
            limit = int(s[pos]) if tight else 1
            result = 0
            
            for digit in range(limit + 1):
                new_bits = bits + (1 if digit == 1 else 0)
                new_tight = tight and (digit == limit)
                new_started = started or (digit == 1)
                
                result = (result + dp(pos + 1, new_bits, new_tight, new_started)) % MOD
            
            return result
        
        return dp(0, 0, True, False)