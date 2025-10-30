class Solution:
    def beautifulNumbers(self, l: int, r: int) -> int:
        from functools import lru_cache

        def count_non_zero(x):
            if x < 1:
                return 0
            s = str(x)
            digits = list(map(int, s))
            n = len(digits)

            @lru_cache(maxsize=None)
            def dp(pos, tight, leading_zero):
                if pos == n:
                    return 0 if leading_zero else 1
                ans = 0
                upper = digits[pos] if tight else 9
                for d in range(0, upper + 1):
                    if d == 0:
                        continue  # Skip zeros as non-zero digits are required
                    new_tight = tight and (d == upper)
                    ans += dp(pos + 1, new_tight, False)
                return ans

            return dp(0, True, True)

        def count_condition_numbers(x):
            if x < 1:
                return 0
            max_sum = len(str(x)) * 9
            res = 0
            for s in range(1, max_sum + 1):
                res += helper(x, s)
            return res

        def helper(y, s):
            if y < 1 or s < 1:
                return 0
            s_str = str(y)
            digits = list(map(int, s_str))
            n = len(digits)
            MOD = s

            @lru_cache(maxsize=None)
            def dp(pos, tight, leading_zero, rem, pm):
                if pos == n:
                    return int(not leading_zero and rem == 0 and pm == 0)
                ans = 0
                upper = digits[pos] if tight else 9
                for d in range(0, upper + 1):
                    new_tight = tight and (d == upper)
                    if leading_zero:
                        if d == 0:
                            ans += dp(pos + 1, new_tight, True, rem, pm)
                        else:
                            if d > rem:
                                continue
                            new_rem = rem - d
                            new_pm = (d % MOD)
                            ans += dp(pos + 1, new_tight, False, new_rem, new_pm)
                    else:
                        if d == 0:
                            continue
                        if rem < d:
                            continue
                        new_rem = rem - d
                        new_pm = (pm * d) % MOD
                        ans += dp(pos + 1, new_tight, False, new_rem, new_pm)
                return ans

            return dp(0, True, True, s, 1 % MOD)

        # Compute count_zero_digits
        total_numbers = r - l + 1
        non_zero_count_r = count_non_zero(r)
        non_zero_count_l_minus_1 = count_non_zero(l - 1) if l > 1 else 0
        count_zero_digits = total_numbers - (non_zero_count_r - non_zero_count_l_minus_1)

        # Compute count_condition
        con_r = count_condition_numbers(r)
        con_l = count_condition_numbers(l - 1) if l > 1 else 0
        count_condition = con_r - con_l

        return count_zero_digits + count_condition