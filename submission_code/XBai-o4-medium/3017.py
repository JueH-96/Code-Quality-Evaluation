class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def count(x):
            if x == 0:
                return 0
            digits = list(map(int, str(x)))
            n = len(digits)
            from functools import lru_cache

            @lru_cache(maxsize=None)
            def dp(pos, tight, started, rem, diff):
                if pos == n:
                    return 1 if (started and rem == 0 and diff == 0) else 0
                res = 0
                max_d = digits[pos] if tight else 9
                for d in range(0, max_d + 1):
                    new_tight = tight and (d == max_d)
                    new_started = started or (d != 0)
                    if not started:
                        if d == 0:
                            new_rem = rem
                            new_diff = diff
                        else:
                            new_rem = d % k
                            new_diff = 1 if d % 2 == 0 else -1
                    else:
                        new_rem = (rem * 10 + d) % k
                        new_diff = diff + (1 if d % 2 == 0 else -1)
                    res += dp(pos + 1, new_tight, new_started, new_rem, new_diff)
                return res

            return dp(0, True, False, 0, 0)
        
        return count(high) - count(low - 1)