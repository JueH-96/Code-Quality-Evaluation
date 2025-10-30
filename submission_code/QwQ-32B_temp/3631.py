class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        MOD = 10**9 + 7
        n = len(s)
        max_c = n  # Maximum possible count of set bits is the length of s

        # Precompute g(c) for all c from 1 to max_c
        g = [0] * (max_c + 1)  # g[0] unused
        g[1] = 0
        for c in range(2, max_c + 1):
            bits = bin(c).count('1')
            g[c] = 1 + g[bits]

        from functools import lru_cache

        @lru_cache(maxsize=None)
        def dp(pos, tight, started, cnt):
            if pos == n:
                # Check if the number is valid (less than n and meets the condition)
                if started and not tight and (g[cnt] <= k - 1):
                    return 1
                else:
                    return 0
            res = 0
            max_bit = int(s[pos]) if tight else 1
            for bit in [0, 1]:
                if bit > max_bit:
                    continue
                new_tight = tight and (bit == max_bit)
                new_started = started or (bit == 1)
                if new_started:
                    new_cnt = cnt + (1 if bit else 0)
                else:
                    new_cnt = 0  # Not started yet, count remains 0
                # Ensure new_cnt doesn't exceed max_c (though it shouldn't)
                res += dp(pos + 1, new_tight, new_started, new_cnt)
                res %= MOD
            return res % MOD

        result = dp(0, True, False, 0)
        return result % MOD