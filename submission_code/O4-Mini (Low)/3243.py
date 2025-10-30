from functools import lru_cache

class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        # Convert suffix s to integer and compute base = 10^k
        k = len(s)
        suf = int(s)
        base = 10 ** k
        
        # Compute allowed y range so that x = y*base + suf is in [start, finish]
        # y >= 0
        # y*base + suf >= start  => y >= ceil((start - suf)/base)
        # y*base + suf <= finish => y <= floor((finish - suf)/base)
        # If finish < suf, no solutions
        if finish < suf:
            return 0
        
        # Ceil division for y_lo
        y_lo = (start - suf + base - 1) // base
        y_hi = (finish - suf) // base
        
        # Enforce y >= 0
        if y_hi < 0:
            return 0
        y_lo = max(y_lo, 0)
        if y_lo > y_hi:
            return 0
        
        # Digit-DP: count numbers y in [0..N] whose every digit <= limit
        def count_upto(N: int) -> int:
            digs = list(map(int, str(N)))
            n = len(digs)
            
            @lru_cache(None)
            def dp(pos: int, tight: bool) -> int:
                # pos: current index in digs, tight: whether prefix equals N's prefix
                if pos == n:
                    # constructed a valid number
                    return 1
                ub = digs[pos] if tight else 9
                total = 0
                # choose digit d at this position
                for d in range(0, min(limit, ub) + 1):
                    total += dp(pos + 1, tight and (d == ub))
                return total
            
            return dp(0, True)
        
        # Answer = count in [0..y_hi] minus count in [0..y_lo-1]
        return count_upto(y_hi) - count_upto(y_lo - 1)