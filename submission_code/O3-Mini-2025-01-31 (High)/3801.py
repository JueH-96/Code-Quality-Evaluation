class Solution:
    def beautifulNumbers(self, l: int, r: int) -> int:
        from functools import lru_cache

        # Helper function: count the number of beautiful numbers in [1, x]
        def count_up_to(x: int) -> int:
            if x < 1:
                return 0
            s = str(x)
            n = len(s)
            digits = [int(ch) for ch in s]

            @lru_cache(maxsize=None)
            def dp(pos, tight, started, has_zero, current_sum, current_product):
                # When we've processed all digit positions:
                if pos == n:
                    # If we never started (all leading zeros) it's not a valid number.
                    if not started:
                        return 0
                    # If the number has at least one zero digit, its product is 0,
                    # and 0 % (positive sum) == 0, so it is beautiful.
                    if has_zero:
                        return 1
                    # Otherwise, for a number with nonzero digits, check the condition:
                    # the product of digits must be divisible by the sum of digits.
                    return 1 if (current_sum > 0 and (current_product % current_sum == 0)) else 0

                res = 0
                # Determine the upper bound for the current digit.
                limit = digits[pos] if tight else 9
                for d in range(0, limit + 1):
                    new_tight = tight and (d == limit)
                    if not started:
                        if d == 0:
                            # Still in the leading zeros.
                            res += dp(pos + 1, new_tight, False, False, 0, 1)
                        else:
                            # Start the number: initialize sum and product with the digit d.
                            res += dp(pos + 1, new_tight, True, False, d, d)
                    else:
                        if d == 0:
                            # Once we've started, if we pick a 0, then regardless of what came before,
                            # the overall product becomes 0 (and remains 0 thereafter). Mark has_zero True.
                            res += dp(pos + 1, new_tight, True, True, current_sum, 0)
                        else:
                            if has_zero:
                                # If a zero has already been encountered, the product is 0.
                                res += dp(pos + 1, new_tight, True, True, current_sum + d, 0)
                            else:
                                # Update the sum and product normally
                                res += dp(pos + 1, new_tight, True, False, current_sum + d, current_product * d)
                return res

            return dp(0, True, False, False, 0, 1)

        # Count beautiful numbers in [l, r] by using the standard technique:
        # count_beautiful(r) - count_beautiful(l - 1).
        return count_up_to(r) - count_up_to(l - 1)