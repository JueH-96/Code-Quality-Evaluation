class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7

        def subtract_one(num_str):
            num = list(num_str)
            i = len(num) - 1
            while i >= 0:
                if num[i] == '0':
                    num[i] = '9'
                    i -= 1
                else:
                    num[i] = str(int(num[i]) - 1)
                    break
            if num[0] == '0' and len(num) > 1:
                num = num[1:]
            return ''.join(num) if num else '0'

        def count_less_or_equal(n_str, s):
            digits = list(map(int, n_str))
            n = len(digits)

            from functools import lru_cache

            @lru_cache(maxsize=None)
            def dp(pos, tight, sum_so_far, leading_zero):
                if pos == n:
                    if leading_zero:
                        return 1 if 0 <= s else 0
                    else:
                        return 1 if sum_so_far <= s else 0
                res = 0
                limit = digits[pos] if tight else 9
                for d in range(0, limit + 1):
                    new_tight = tight and (d == limit)
                    new_leading_zero = leading_zero and (d == 0)
                    new_sum = sum_so_far
                    if not new_leading_zero:
                        if leading_zero:
                            new_sum = d
                        else:
                            new_sum = sum_so_far + d
                    if new_sum > s:
                        continue
                    res += dp(pos + 1, new_tight, new_sum, new_leading_zero)
                    res %= MOD
                return res % MOD

            return dp(0, True, 0, True)

        if num1 == '1':
            num1_minus_1 = '0'
        else:
            num1_minus_1 = subtract_one(num1)

        a = count_less_or_equal(num2, max_sum)
        b = count_less_or_equal(num2, min_sum - 1)
        c = count_less_or_equal(num1_minus_1, max_sum)
        d = count_less_or_equal(num1_minus_1, min_sum - 1)

        result = (a - b - c + d) % MOD
        return result if result >= 0 else result + MOD