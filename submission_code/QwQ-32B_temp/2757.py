class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7

        def subtract_one(s):
            s_list = list(s)
            i = len(s_list) - 1
            while i >= 0 and s_list[i] == '0':
                s_list[i] = '9'
                i -= 1
            if i < 0:
                return '0'
            s_list[i] = str(int(s_list[i]) - 1)
            new_s = ''.join(s_list).lstrip('0')
            return new_s if new_s else '0'

        def count_less_or_equal(s):
            digits = list(map(int, s))
            n = len(digits)
            from functools import lru_cache

            @lru_cache(maxsize=None)
            def dp(pos, tight, current_sum):
                if pos == n:
                    return 1 if (min_sum <= current_sum <= max_sum) else 0
                limit = digits[pos] if tight else 9
                total = 0
                for d in range(0, limit + 1):
                    new_tight = tight and (d == limit)
                    new_sum = current_sum + d
                    total += dp(pos + 1, new_tight, new_sum)
                return total

            return dp(0, True, 0)

        num1_minus_1 = subtract_one(num1)
        ans = (count_less_or_equal(num2) - count_less_or_equal(num1_minus_1)) % MOD
        return ans