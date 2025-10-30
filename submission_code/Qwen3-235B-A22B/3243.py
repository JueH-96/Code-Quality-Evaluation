class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        from functools import lru_cache

        def count_valid(Y_str, P, limit_inner):
            digits = list(map(int, Y_str))
            if len(digits) != P:
                return 0

            @lru_cache(maxsize=None)
            def dp(pos, tight):
                if pos == P:
                    return 1
                res = 0
                current_digit = digits[pos]
                max_d = min(current_digit, limit_inner) if tight else limit_inner

                if pos == 0:
                    if max_d < 1:
                        return 0
                    low_d = 1
                else:
                    low_d = 0

                for d in range(low_d, max_d + 1):
                    new_tight = tight and (d == current_digit)
                    res += dp(pos + 1, new_tight)
                return res

            return dp(0, True)

        s_num = int(s)
        L = len(s)
        ans = 0

        # Handle P=0 case
        if start <= s_num <= finish:
            ans += 1

        # Determine the maximum number of digits possible for N
        try:
            max_num_digits = len(str(finish))
        except:
            max_num_digits = len(str(finish))
        max_p = max(0, max_num_digits - L)

        for P in range(1, max_p + 1):
            X_min_val = 10 ** (P - 1)
            X_max_val = (10 ** P) - 1
            denominator = 10 ** L

            # Calculate X_low_val
            numerator_low = start - s_num
            if numerator_low <= 0:
                X_low_val = 0
            else:
                X_low_val = (numerator_low + denominator - 1) // denominator
            X_low = max(X_min_val, X_low_val)

            # Calculate X_high_val
            numerator_high = finish - s_num
            if numerator_high < 0:
                X_high_val = -1
            else:
                X_high_val = numerator_high // denominator
            X_high = min(X_max_val, X_high_val)

            if X_high < X_low:
                continue

            # Compute upper_count
            Y_str_upper = str(X_high)
            if len(Y_str_upper) != P:
                upper_count = 0
            else:
                upper_count = count_valid(Y_str_upper, P, limit)

            # Compute lower_count
            if X_low == 0:
                lower_count = 0
            else:
                Y_low = X_low - 1
                Y_str_lower = str(Y_low)
                if len(Y_str_lower) != P:
                    lower_count = 0
                else:
                    lower_count = count_valid(Y_str_lower, P, limit)

            ans += max(0, upper_count - lower_count)

        return ans