class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10**9 + 7

        def subtract_one(s):
            s_list = list(s)
            i = len(s_list) - 1
            while i >= 0:
                if s_list[i] == '0':
                    s_list[i] = '9'
                    i -= 1
                else:
                    s_list[i] = str(int(s_list[i]) - 1)
                    break
            new_str = ''.join(s_list).lstrip('0')
            return new_str if new_str else '0'

        def count_stepping(s):
            len_s = len(s)
            if len_s == 0:
                return 0

            max_k = len_s - 1
            step1 = 0
            if max_k >= 1:
                dp_step1 = [[0] * 10 for _ in range(max_k + 2)]

                for d in range(1, 10):
                    dp_step1[1][d] = 1

                for k in range(2, max_k + 1):
                    for d in range(10):
                        for p in range(10):
                            if abs(d - p) == 1:
                                dp_step1[k][d] += dp_step1[k - 1][p]
                                dp_step1[k][d] %= MOD

                for k in range(1, max_k + 1):
                    step1 += sum(dp_step1[k])
                    step1 %= MOD

            s_digits = list(s)
            n = len(s_digits)

            from functools import lru_cache
            @lru_cache(maxsize=None)
            def dp(pos, prev_digit, tight):
                if pos == n:
                    return 1
                limit = int(s_digits[pos]) if tight else 9
                total = 0
                for d in range(0, limit + 1):
                    new_tight = tight and (d == limit)
                    if pos == 0:
                        if d == 0:
                            continue
                        total += dp(pos + 1, d, new_tight)
                        total %= MOD
                    else:
                        if abs(d - prev_digit) == 1:
                            total += dp(pos + 1, d, new_tight)
                            total %= MOD
                return total % MOD

            step2 = dp(0, -1, True) if len_s >= 1 else 0
            return (step1 + step2) % MOD

        low_minus_1 = subtract_one(low)
        count_high = count_stepping(high)
        count_low_minus_1 = count_stepping(low_minus_1) if low != '1' else 0
        return (count_high - count_low_minus_1) % MOD