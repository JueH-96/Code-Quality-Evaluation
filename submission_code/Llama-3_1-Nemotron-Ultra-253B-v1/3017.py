class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def count(n):
            if n < 0:
                return 0
            digits = list(map(int, str(n)))
            from functools import lru_cache

            @lru_cache(maxsize=None)
            def dp(pos, tight, diff, mod, started):
                if pos == len(digits):
                    return 1 if (started and diff == 0 and mod == 0) else 0
                res = 0
                max_d = digits[pos] if tight else 9
                for d in range(0, max_d + 1):
                    new_tight = tight and (d == max_d)
                    new_started = started or (d != 0)
                    new_diff = diff
                    new_mod = mod

                    if new_started:
                        if d % 2 == 0:
                            new_diff += 1
                        else:
                            new_diff -= 1
                        if started:
                            new_mod = (mod * 10 + d) % k
                        else:
                            new_mod = d % k
                    else:
                        new_mod = 0

                    res += dp(pos + 1, new_tight, new_diff, new_mod, new_started)
                return res

            return dp(0, True, 0, 0, False)

        return count(high) - count(low - 1)