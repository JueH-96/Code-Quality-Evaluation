class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def count(n):
            if n == 0:
                return 0
            digits = list(map(int, str(n)))
            n_digits = len(digits)
            from functools import lru_cache

            @lru_cache(None)
            def dp(pos, mod, diff, tight, leading_zero):
                if pos == n_digits:
                    return int(not leading_zero and mod == 0 and diff == 0)
                res = 0
                up = digits[pos] if tight else 9
                for d in range(0, up + 1):
                    new_tight = tight and (d == up)
                    new_mod = (mod * 10 + d) % k
                    if leading_zero:
                        if d == 0:
                            new_leading_zero = True
                            new_diff = diff
                        else:
                            new_leading_zero = False
                            if d % 2 == 0:
                                new_diff = 1
                            else:
                                new_diff = -1
                    else:
                        new_leading_zero = False
                        if d % 2 == 0:
                            new_diff = diff + 1
                        else:
                            new_diff = diff - 1
                    res += dp(pos + 1, new_mod, new_diff, new_tight, new_leading_zero)
                return res
            return dp(0, 0, 0, True, True)
        return count(high) - count(low - 1)