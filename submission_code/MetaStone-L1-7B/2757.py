class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7

        def subtract_one(s):
            s_list = list(s)
            i = len(s_list) - 1
            while i >= 0:
                if s_list[i] != '0':
                    s_list[i] = str(int(s_list[i]) - 1)
                    break
                else:
                    s_list[i] = '9'
                    i -= 1
            # Remove leading zeros
            first_non_zero = 0
            while first_non_zero < len(s_list) and s_list[first_non_zero] == '0':
                first_non_zero += 1
            if first_non_zero == len(s_list):
                return '0'
            s_list = s_list[first_non_zero:]
            if not s_list:
                return '0'
            return ''.join(s_list)

        def helper(X):
            digits = list(map(int, X))
            n = len(digits)

            from functools import lru_cache

            @lru_cache(maxsize=None)
            def dp(pos, sum_so_far, tight, leading_zero):
                if pos == n:
                    if sum_so_far >= min_sum and sum_so_far <= max_sum:
                        if leading_zero:
                            return 1 if (min_sum == 0 and max_sum == 0) else 0
                        else:
                            return 1
                else:
                    total = 0
                    limit = digits[pos] if tight else 9
                    for d in range(0, limit + 1):
                        new_tight = tight and (d == limit)
                        new_leading_zero = leading_zero and (d == 0)
                        new_sum = sum_so_far
                        if leading_zero:
                            if d == 0:
                                new_sum = sum_so_far
                            else:
                                new_sum = d
                        else:
                            new_sum = sum_so_far + d
                        res = dp(pos + 1, new_sum, new_tight, new_leading_zero)
                        total += res
                        total %= MOD
                    return total % MOD

            result = dp(0, 0, True, True)
            return result

        f_num2 = helper(num2)
        num1_minus_1 = subtract_one(num1)
        f_num1_minus_1 = helper(num1_minus_1)
        ans = (f_num2 - f_num1_minus_1) % MOD
        return ans if ans >= 0 else ans + MOD