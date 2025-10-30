from functools import lru_cache

MOD = 1_000_000_007     # result is required modulo 1e9+7

class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:

        # ------------------------------------------------------------------ #
        # helper :  subtract 1 from a non-negative decimal string            #
        # ------------------------------------------------------------------ #
        def minus_one(num: str) -> str:
            # num is a decimal string without leading zeros, num >= "0"
            if num == "0":
                return "0"
            s = list(num)
            i = len(s) - 1
            # borrow
            while i >= 0 and s[i] == '0':
                s[i] = '9'
                i -= 1
            s[i] = str(int(s[i]) - 1)
            res = ''.join(s).lstrip('0')
            return res if res else "0"

        # ------------------------------------------------------------------ #
        # helper :  count stepping numbers in [1, bound]                     #
        # ------------------------------------------------------------------ #
        def count_up_to(bound: str) -> int:
            n = len(bound)

            @lru_cache(None)
            def dfs(pos: int, prev: int, tight: int, started: int) -> int:
                """
                pos     : current index in the string (0 … n)
                prev    : previous digit (0‐9) or 10 meaning “none yet”
                tight   : 1 if prefix equals bound so far, otherwise 0
                started : 1 if we have placed first non-zero digit, otherwise 0
                """
                if pos == n:                     # all positions processed
                    return 1 if started else 0   # valid number iff started

                limit = int(bound[pos]) if tight else 9
                res = 0

                for d in range(limit + 1):
                    tight2 = tight and (d == limit)

                    if started == 0 and d == 0:
                        # still skipping leading zeros
                        res += dfs(pos + 1, 10, tight2, 0)
                    else:
                        if started == 0:
                            # first non-zero digit (cannot be 0)
                            res += dfs(pos + 1, d, tight2, 1)
                        else:
                            # have previous digit → stepping constraint
                            if abs(d - prev) == 1:
                                res += dfs(pos + 1, d, tight2, 1)

                    if res >= MOD:               # avoid big integers
                        res %= MOD

                return res % MOD

            return dfs(0, 10, 1, 0)

        # ------------------------------------------------------------------ #
        # main logic : answer = f(high) − f(low − 1)                         #
        # ------------------------------------------------------------------ #
        high_cnt = count_up_to(high)
        low_minus_one = minus_one(low)
        low_cnt  = count_up_to(low_minus_one) if low_minus_one != "0" else 0

        return (high_cnt - low_cnt) % MOD