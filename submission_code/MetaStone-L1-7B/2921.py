def countSteppingNumbers(low, high):
    MOD = 10**9 + 7

    def count_upto(s):
        if s == "0":
            return 0
        n = len(s)
        total = 0

        # Count numbers with length less than n
        for l in range(1, n):
            if l == 1:
                total += 9
            else:
                total += count_same_length(l)
            total %= MOD

        # Count numbers with same length as s and <= s
        same_length = count_same_length_l(s, n)
        total = (total + same_length) % MOD
        return total

    def count_same_length(l):
        if l == 1:
            return 9
        dp = [{} for _ in range(l)]
        for d in range(1, 10):
            dp[0][d] = 1

        for i in range(1, l):
            for d_prev in dp[i-1]:
                for delta in (-1, 1):
                    d_current = d_prev + delta
                    if 0 <= d_current <= 9:
                        if d_current in dp[i]:
                            dp[i][d_current] += dp[i-1][d_prev]
                        else:
                            dp[i][d_current] = dp[i-1][d_prev]
        return sum(dp[l-1].values()) % MOD

    def count_same_length_l(n_str, l):
        if l != len(n_str):
            return 0
        digits = list(map(int, n_str))
        n = len(digits)
        from functools import lru_cache

        @lru_cache(maxsize=None)
        def dp(pos, prev, tight):
            if pos == n:
                return 1
            limit = digits[pos] if tight else 9
            total = 0
            for d in range(0, limit + 1):
                if pos == 0 and d == 0:
                    continue
                if abs(d - prev) != 1:
                    continue
                new_tight = tight and (d == limit)
                total += dp(pos + 1, d, new_tight)
            return total % MOD

        return dp(0, -1, 1) % MOD

    res = (count_upto(high) - count_upto(low - 1)) % MOD
    return res if res >= 0 else res + MOD