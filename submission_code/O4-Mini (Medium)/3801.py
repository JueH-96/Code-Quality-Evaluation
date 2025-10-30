from functools import lru_cache

class Solution:
    def beautifulNumbers(self, l: int, r: int) -> int:
        def count_upto(X: int) -> int:
            # count beautiful numbers in [1..X]
            if X <= 0:
                return 0
            ds = list(map(int, str(X)))
            n = len(ds)
            total = 0
            # maximum possible digit sum for n digits is 9*n
            maxS = 9 * n
            for S in range(1, maxS + 1):
                # dp over position, sum so far, product mod S, started flag, tight flag
                @lru_cache(None)
                def dfs(i: int, sumd: int, mod: int, started: bool, tight: bool) -> int:
                    # prune sums exceeding S
                    if sumd > S:
                        return 0
                    # end of digits
                    if i == n:
                        # must have exactly sum S, product mod S == 0, and at least one non-zero digit
                        return 1 if (started and sumd == S and mod == 0) else 0
                    res = 0
                    up = ds[i] if tight else 9
                    for d in range(up + 1):
                        ntight = tight and (d == up)
                        nstarted = started or (d != 0)
                        # new digit sum: only add if it's a real digit (or it's zero but started already)
                        nsum = sumd + (d if nstarted else 0)
                        if nsum > S:
                            continue
                        # new product mod S
                        if not nstarted:
                            # still leading zeros, product stays as before
                            nmod = mod
                        else:
                            # multiply by d
                            nmod = (mod * d) % S
                        res += dfs(i + 1, nsum, nmod, nstarted, ntight)
                    return res

                # initial product is 1, no digits started, tight=True
                total += dfs(0, 0, 1, False, True)
                dfs.cache_clear()
            return total

        return count_upto(r) - count_upto(l - 1)