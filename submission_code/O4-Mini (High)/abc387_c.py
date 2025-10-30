def main():
    import sys
    sys.setrecursionlimit(10000)
    data = sys.stdin.read().strip().split()
    if not data:
        return
    L, R = map(int, data)

    def count_snakes(x):
        # Count Snake numbers in [10, x]
        s = str(x)
        digits = list(map(int, s))
        n = len(digits)
        from functools import lru_cache

        @lru_cache(None)
        def dfs(pos, tight, started, top, maxo, cnt):
            # pos: current index in digits [0..n)
            # tight: whether prefix == x's prefix
            # started: whether we've placed the top digit (first non-zero)
            # top: the top digit value (0..9) if started, else 0
            # maxo: the current maximum of other digits seen so far
            # cnt: count of digits placed after started (including top digit)
            if pos == n:
                # At end, valid if we have at least 2 digits and
                # top > every other digit (i.e. top > maxo)
                return 1 if (started and cnt >= 2 and top > maxo) else 0

            res = 0
            limit = digits[pos] if tight else 9
            for d in range(limit + 1):
                nt = tight and (d == limit)
                if not started:
                    # still skipping leading zeros
                    if d == 0:
                        # remain not started
                        res += dfs(pos + 1, nt, False, 0, 0, 0)
                    else:
                        # start the number: this digit is top
                        res += dfs(pos + 1, nt, True, d, 0, 1)
                else:
                    # already started: update maxo and count
                    new_maxo = maxo if maxo > d else d
                    res += dfs(pos + 1, nt, True, top, new_maxo, cnt + 1)
            return res

        return dfs(0, True, False, 0, 0, 0)

    # Compute answer as F(R) - F(L-1). For L>=10 always so L-1>=9 => no negative.
    ans = count_snakes(R) - count_snakes(L - 1)
    sys.stdout.write(str(ans))


if __name__ == "__main__":
    main()