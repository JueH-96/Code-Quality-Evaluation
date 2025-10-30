class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        from functools import lru_cache
        
        # how many beautiful numbers are ≤ limit
        def count(limit: int) -> int:
            if limit <= 0:
                return 0
            digits = list(map(int, str(limit)))
            n = len(digits)
            
            @lru_cache(maxsize=None)
            def dfs(pos: int, mod: int, diff: int, tight: int, started: int) -> int:
                """
                pos     : current position in the digit list
                mod     : value (mod k) of the number built so far
                diff    : (#even digits) − (#odd digits) seen so far
                tight   : 1 if the prefix equals the limit’s prefix, else 0
                started : 1 if we have already placed a non-leading-zero digit
                """
                if pos == n:                          # finished all positions
                    return 1 if started and diff == 0 and mod == 0 else 0
                
                upper = digits[pos] if tight else 9
                total = 0
                for d in range(upper + 1):
                    ntight   = tight and (d == upper)
                    nstarted = started or d != 0
                    
                    if nstarted:
                        prev_mod = mod if started else 0      # before first real digit value is 0
                        nmod  = (prev_mod * 10 + d) % k
                        ndiff = diff + (1 if d % 2 == 0 else -1)
                    else:
                        nmod  = 0        # still have no digits => value 0
                        ndiff = diff     # diff unchanged
                    total += dfs(pos + 1, nmod, ndiff, int(ntight), int(nstarted))
                return total
            
            return dfs(0, 0, 0, 1, 0)
        
        # beautiful numbers in [low, high] = ≤high minus ≤(low-1)
        return count(high) - count(low - 1)