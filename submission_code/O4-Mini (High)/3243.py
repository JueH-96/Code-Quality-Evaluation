class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        from functools import lru_cache

        # Convert suffix string to integer and its power of ten
        t = int(s)
        L = len(s)
        m = 10 ** L

        # Compute the range of possible prefixes P such that
        # x = P * 10^L + t lies in [start, finish].
        # Ceil((start - t) / m)
        minP = (start - t + m - 1) // m
        if minP < 0:
            minP = 0
        # Floor((finish - t) / m)
        maxP = (finish - t) // m
        if maxP < minP:
            return 0

        # Digit-DP: count how many integers P in [0..N] have all digits <= limit
        def count_valid_prefixes(N: int) -> int:
            if N < 0:
                return 0
            digs = list(map(int, str(N)))
            n = len(digs)

            @lru_cache(None)
            def dp(i: int, tight: bool) -> int:
                # If we've assigned all digits, that's one valid number
                if i == n:
                    return 1
                res = 0
                # If we're tight, we can't exceed digs[i]; otherwise up to 9
                up = digs[i] if tight else 9
                # But we also must respect the 'limit' on each digit
                up = min(up, limit)
                for d in range(up + 1):
                    res += dp(i + 1, tight and (d == digs[i]))
                return res

            return dp(0, True)

        # Answer is count of P in [minP..maxP]
        return count_valid_prefixes(maxP) - count_valid_prefixes(minP - 1)