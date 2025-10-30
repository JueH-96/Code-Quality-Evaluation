class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        from functools import lru_cache

        def count(x):
            digits = list(map(int, str(x)))
            n = len(digits)

            @lru_cache(maxsize=None)
            def dp(pos, diff, mod, tight, started):
                if pos == n:
                    return int(started and diff == 0 and mod == 0)
                
                total = 0
                limit = digits[pos] if tight else 9
                
                for d in range(0, limit + 1):
                    new_tight = tight and (d == limit)
                    new_started = started or d != 0
                    new_diff = diff
                    new_mod = mod
                    if new_started:
                        if d % 2 == 0:
                            new_diff += 1
                        else:
                            new_diff -= 1
                        new_mod = (mod * 10 + d) % k
                    total += dp(pos + 1, new_diff, new_mod, new_tight, new_started)
                
                return total

            return dp(0, 0, 0, True, False)

        return count(high) - count(low - 1)