class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10**9 + 7

        def subtract_one(s):
            s_list = list(s)
            i = len(s_list) - 1
            while i >= 0 and s_list[i] == '0':
                s_list[i] = '9'
                i -= 1
            if i < 0:
                return "0"
            s_list[i] = str(int(s_list[i]) - 1)
            res = ''.join(s_list).lstrip('0')
            return res if res else "0"

        def count_stepping_up_to(N_str):
            if not N_str:
                return 0
            digits = list(map(int, N_str))
            n = len(digits)
            from functools import lru_cache

            @lru_cache(maxsize=None)
            def dp(pos, prev_digit, is_tight, started):
                if pos == n:
                    return 1 if started else 0
                res = 0
                max_d = digits[pos] if is_tight else 9
                for d in range(0, max_d + 1):
                    new_is_tight = is_tight and (d == max_d)
                    if not started:
                        if d == 0:
                            res += dp(pos + 1, prev_digit, new_is_tight, False)
                        else:
                            res += dp(pos + 1, d, new_is_tight, True)
                    else:
                        if abs(d - prev_digit) == 1:
                            res += dp(pos + 1, d, new_is_tight, True)
                return res % MOD

            total = dp(0, 0, True, False)
            return total % MOD

        count_high = count_stepping_up_to(high)
        low_minus_1 = subtract_one(low)
        count_low = count_stepping_up_to(low_minus_1)
        ans = (count_high - count_low) % MOD
        return ans