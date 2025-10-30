class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        from functools import lru_cache

        # Returns count of beautiful integers in [1, int(s)]
        def dp_count(s: str) -> int:
            n = len(s)

            @lru_cache(None)
            def dfs(pos: int, diff: int, rem: int, tight: bool, leading_zero: bool) -> int:
                # pos: current digit index
                # diff: (#even digits) - (#odd digits) so far
                # rem: current value modulo k
                # tight: whether prefix matches s so far
                # leading_zero: still haven't placed a non-zero digit
                if pos == n:
                    # at end, count if we have at least one digit, diff == 0 and divisible by k
                    return int((not leading_zero) and diff == 0 and rem == 0)

                res = 0
                limit = int(s[pos]) if tight else 9
                for d in range(limit + 1):
                    ntight = tight and (d == limit)
                    nleading = leading_zero and (d == 0)
                    nrem = (rem * 10 + d) % k

                    ndiff = diff
                    # only update diff once we've placed the first non-zero digit (or if d!=0 after that)
                    if not nleading:
                        # 0 is even
                        if d % 2 == 0:
                            ndiff += 1
                        else:
                            ndiff -= 1

                    res += dfs(pos + 1, ndiff, nrem, ntight, nleading)
                return res

            return dfs(0, 0, 0, True, True)

        # Count in [low, high] = count( [1,high] ) - count( [1, low-1] )
        return dp_count(str(high)) - dp_count(str(low - 1))