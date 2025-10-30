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
                return '0'  # This case shouldn't occur as per problem constraints
            s_list[i] = str(int(s_list[i]) - 1)
            if s_list[0] == '0' and len(s_list) > 1:
                return ''.join(s_list[1:])
            return ''.join(s_list)

        low_minus_1 = subtract_one(low)

        def count_stepping(s):
            m = len(s)
            sum_less = 0

            # Sum of stepping numbers with length < m
            for l in range(1, m):
                sum_less += sum_stepping_numbers_length_l(l)
                sum_less %= MOD

            # Sum of stepping numbers with length m and <= s
            sum_current = count_stepping_length_m(s)
            return (sum_less + sum_current) % MOD

        def sum_stepping_numbers_length_l(target_length):
            if target_length == 0:
                return 0
            dp_prev = {d: 1 for d in range(1, 10)}
            if target_length == 1:
                return sum(dp_prev.values()) % MOD
            for current_length in range(2, target_length + 1):
                dp_current = {}
                for prev_d in dp_prev:
                    next_d1 = prev_d - 1
                    next_d2 = prev_d + 1
                    if next_d1 >= 0:
                        dp_current[next_d1] = (dp_current.get(next_d1, 0) + dp_prev[prev_d]) % MOD
                    if next_d2 <= 9:
                        dp_current[next_d2] = (dp_current.get(next_d2, 0) + dp_prev[prev_d]) % MOD
                dp_prev = dp_current
            return sum(dp_prev.values()) % MOD

        def count_stepping_length_m(s):
            digits = list(map(int, s))
            m = len(digits)

            from functools import lru_cache

            @lru_cache(maxsize=None)
            def dfs(i, prev, is_tight):
                if i == m:
                    return 1
                total = 0
                if i == 0:
                    max_d = digits[0] if is_tight else 9
                    for d in range(1, max_d + 1):
                        new_is_tight = is_tight and (d == digits[0])
                        for next_d in [d - 1, d + 1]:
                            if 0 <= next_d <= 9:
                                total += dfs(i + 1, d, new_is_tight)
                                total %= MOD
                else:
                    for next_d in [prev - 1, prev + 1]:
                        if 0 <= next_d <= 9:
                            if is_tight:
                                if next_d > digits[i]:
                                    continue
                                new_is_tight = (next_d == digits[i])
                            else:
                                new_is_tight = False
                            total += dfs(i + 1, next_d, new_is_tight)
                            total %= MOD
                return total % MOD

            return dfs(0, None, True) % MOD

        count_high = count_stepping(high)
        count_low_minus_1 = count_stepping(low_minus_1)
        return (count_high - count_low_minus_1) % MOD