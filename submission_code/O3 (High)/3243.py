from functools import lru_cache

class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        """
        Count integers x in [start, finish] that
        1. end with the fixed suffix `s`
        2. every decimal digit of x does not exceed `limit`
        """

        suf_len = len(s)                # length of the suffix
        suf_val = int(s)                # numeric value of the suffix
        base    = 10 ** suf_len         # 10^|s|

        # largest / smallest prefix value that can appear
        hi = (finish - suf_val) // base
        if hi < 0:                           # even the smallest possible number is too big
            return 0

        lo = (start - suf_val + base - 1) // base   # ceil((start - suf_val)/base)
        if lo < 0:
            lo = 0

        if lo > hi:
            return 0

        # amount of prefixes in [0 .. hi]   – prefixes in [0 .. lo-1]
        return self._count_prefixes(hi, limit) - self._count_prefixes(lo - 1, limit)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _count_prefixes(self, n: int, limit: int) -> int:
        """
        How many integers p with 0 <= p <= n have every digit <= limit
        """
        if n < 0:
            return 0
        digits = list(map(int, str(n)))
        m = len(digits)

        @lru_cache(maxsize=None)
        def dfs(pos: int, tight: bool) -> int:
            # pos  : current index in `digits`
            # tight: the prefix so far is equal to the prefix of n
            if pos == m:
                return 1     # formed a valid number

            upper = digits[pos] if tight else 9
            upper = min(upper, limit)        # digits larger than `limit` are forbidden

            total = 0
            for d in range(0, upper + 1):
                total += dfs(pos + 1, tight and (d == digits[pos]))
            return total

        return dfs(0, True)