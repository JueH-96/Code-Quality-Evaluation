from functools import lru_cache

class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        # Count beautiful numbers <= bound
        def count_up_to(bound: int) -> int:
            s = list(map(int, str(bound)))
            n = len(s)
            
            @lru_cache(None)
            def dfs(pos: int, mod: int, diff: int, tight: bool, started: bool) -> int:
                if pos == n:
                    # valid if we've started (non-zero number), mod == 0, and even_count == odd_count => diff == 0
                    return int(started and mod == 0 and diff == 0)
                
                limit = s[pos] if tight else 9
                total = 0
                for d in range(0, limit + 1):
                    ntight = tight and (d == limit)
                    nstarted = started or (d != 0)
                    # update mod always (leading zeros effectively do nothing to mod)
                    nmod = (mod * 10 + d) % k
                    # update diff only if we've started (so we don't count leading zeros)
                    ndiff = diff
                    if nstarted:
                        # If we just started with this digit, or were already started, count evens/odds
                        if d % 2 == 0:
                            ndiff += 1
                        else:
                            ndiff -= 1
                    # Recurse
                    total += dfs(pos + 1, nmod, ndiff, ntight, nstarted)
                return total
            
            return dfs(0, 0, 0, True, False)
        
        # Count beautiful in [low, high] = count_up_to(high) - count_up_to(low - 1)
        return count_up_to(high) - count_up_to(low - 1)