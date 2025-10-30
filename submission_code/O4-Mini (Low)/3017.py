from functools import lru_cache

class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        # Digit DP to count numbers <= bound that are divisible by k
        # and have equal even and odd digits.
        
        def count_beautiful(bound: int) -> int:
            s = list(map(int, str(bound)))
            n = len(s)
            
            @lru_cache(None)
            def dp(pos: int, rem: int, diff: int, tight: bool, started: bool) -> int:
                # pos: current index in digits
                # rem: current remainder mod k
                # diff: even_count - odd_count (offset by 0, can be negative)
                # tight: if prefix matches bound
                # started: if we've placed a non-leading-zero digit
                if pos == n:
                    # at end, must have started, rem == 0, diff == 0
                    return int(started and rem == 0 and diff == 0)
                
                limit = s[pos] if tight else 9
                total = 0
                for d in range(limit + 1):
                    ntight = tight and (d == limit)
                    nstarted = started or (d != 0)
                    nrem = (rem * 10 + d) % k if nstarted else 0
                    ndiff = diff
                    if nstarted:
                        # count even/odd
                        if d % 2 == 0:
                            ndiff += 1
                        else:
                            ndiff -= 1
                    total += dp(pos + 1, nrem, ndiff, ntight, nstarted)
                return total
            
            return dp(0, 0, 0, True, False)
        
        # Result is numbers <= high minus numbers < low
        return count_beautiful(high) - count_beautiful(low - 1)