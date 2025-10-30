class Solution:
    MOD = 10**9 + 7

    def countSteppingNumbers(self, low: str, high: str) -> int:
        def subtract_one(s):
            s_list = list(s)
            i = len(s_list) - 1
            while i >= 0 and s_list[i] == '0':
                s_list[i] = '9'
                i -= 1
            if i < 0:
                return '0'
            s_list[i] = str(int(s_list[i]) - 1)
            if s_list[0] == '0' and len(s_list) > 1:
                return ''.join(s_list[1:])
            return ''.join(s_list)

        def count_stepping(k):
            if k == 0:
                return 0
            dp_prev = [0] * 10
            for d in range(1, 10):
                dp_prev[d] = 1
            for i in range(2, k + 1):
                dp_curr = [0] * 10
                for d in range(10):
                    if d > 0:
                        dp_curr[d] += dp_prev[d - 1]
                    if d < 9:
                        dp_curr[d] += dp_prev[d + 1]
                    dp_curr[d] %= self.MOD
                dp_prev = dp_curr
            return sum(dp_prev) % self.MOD

        def f(X):
            len_x = len(X)
            part1 = 0
            for k in range(1, len_x):
                part1 = (part1 + count_stepping(k)) % self.MOD
            if len_x == 0:
                return 0
            digits_x = [int(c) for c in X]
            from functools import lru_cache

            @lru_cache(maxsize=None)
            def dp(pos, prev_digit, is_tight):
                if pos == len_x:
                    return 1
                max_d = digits_x[pos] if is_tight else 9
                possible = []
                if prev_digit - 1 >= 0:
                    possible.append(prev_digit - 1)
                if prev_digit + 1 <= 9:
                    possible.append(prev_digit + 1)
                res = 0
                for d in possible:
                    if d > max_d:
                        continue
                    new_is_tight = is_tight and (d == max_d)
                    res += dp(pos + 1, d, new_is_tight)
                    res %= self.MOD
                return res % self.MOD

            part2 = 0
            for d in range(1, 10):
                if d > digits_x[0]:
                    continue
                part2 = (part2 + dp(1, d, d == digits_x[0])) % self.MOD
            return (part1 + part2) % self.MOD

        low_minus_1 = subtract_one(low)
        f_high = f(high)
        f_low_minus_1 = f(low_minus_1) if low != "1" else 0
        return (f_high - f_low_minus_1) % self.MOD