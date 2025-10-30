# The code provided within the Solution class will be used.
# Helper function and precomputed data should be outside the class methods
# but can be defined within the same script scope.

# Helper function to get exponent of prime p in n
def get_exponent(n, p):
    if n == 0: return 0 # Should not be called with n=0 for digits > 0
    count = 0
    while n > 0 and n % p == 0:
        count += 1
        n //= p
    return count

# Precompute required exponents for allowed sums
# Sums S where prime factors are only 2, 3, 5, 7
allowed_sum_req = {}
# Max sum for a number < 10^9 is 9*9 = 81.
for S in range(1, 82):
    temp_s = S
    req_exp2 = 0
    req_exp3 = 0
    req_exp5 = 0
    req_exp7 = 0

    # Check for factors 2, 3, 5, 7
    while temp_s > 0 and temp_s % 2 == 0:
        req_exp2 += 1
        temp_s //= 2
    while temp_s > 0 and temp_s % 3 == 0:
        req_exp3 += 1
        temp_s //= 3
    while temp_s > 0 and temp_s % 5 == 0:
        req_exp5 += 1
        temp_s //= 5
    while temp_s > 0 and temp_s % 7 == 0:
        req_exp7 += 1
        temp_s //= 7

    if temp_s == 1: # All factors were 2, 3, 5, or 7
        allowed_sum_req[S] = (req_exp2, req_exp3, req_exp5, req_exp7)

# Capping exponents state based on max required exponents (6, 4, 2, 2)
# Choose caps slightly larger than max required.
EXP2_CAP = 7 # Needs to be >= 6
EXP3_CAP = 5 # Needs to be >= 4
EXP5_CAP = 3 # Needs to be >= 2
EXP7_CAP = 3 # Needs to be >= 2


# Precompute exponents of prime factors 2, 3, 5, 7 for non-zero digits {1..9}
# Digit 0 does not contribute factors to the product of non-zero digits.
digit_exponents = {}
for d in range(10):
    if d == 0:
        digit_exponents[d] = (0, 0, 0, 0)
    else:
         digit_exponents[d] = (
            get_exponent(d, 2),
            get_exponent(d, 3),
            get_exponent(d, 5),
            get_exponent(d, 7)
        )

class Solution:
    def beautifulNumbers(self, l: int, r: int) -> int:

        def count_beautiful_up_to(num_str):
            n = len(num_str)
            memo = {}

            # dp(idx, current_sum_non_leading, exp2, exp3, exp5, exp7, tight, is_leading_zero, has_placed_zero)
            # current_sum_non_leading: sum of digits placed so far, excluding leading zeros
            # exp2, etc: capped exponents of 2,3,5,7 in product of *non-zero* digits placed so far (excluding leading zeros)
            # tight: boolean, True if we are restricted by the digits of num_str
            # is_leading_zero: boolean, True if we are currently placing leading zeros
            # has_placed_zero: boolean, True if digit 0 was placed *after* is_leading_zero became False
            def dp(idx, current_sum_non_leading, exp2, exp3, exp5, exp7, tight, is_leading_zero, has_placed_zero):
                if idx == n:
                    # Base case: finished building a number
                    # If is_leading_zero is true, we haven't formed a positive number yet (e.g., empty string or "00").
                    if is_leading_zero:
                         return 0
                    # If sum of non-leading digits is 0, it means the number is 0. Not positive.
                    # This check is redundant if !is_leading_zero and n > 0, as at least one non-zero digit
                    # must have been placed to exit is_leading_zero, making sum > 0.
                    # However, it correctly handles the case where num_str was "0".
                    if current_sum_non_leading == 0:
                         return 0

                    # We have a positive integer with sum of non-leading digits S = current_sum_non_leading.
                    # Let the product of its digits be P. We need P % S == 0.

                    if has_placed_zero:
                        # If a 0 digit was placed after leading zeros, P is 0.
                        # 0 % S == 0 is true for S != 0. We already checked S != 0 above.
                        return 1
                    else:
                        # No 0 digit was placed after the first non-zero digit.
                        # All digits are non-zero. The product P is the product of these digits.
                        # exp2, exp3, exp5, exp7 represent the exponents of P.

                        # Check if S allows for product divisibility by only 2,3,5,7 factors
                        if current_sum_non_leading not in allowed_sum_req:
                            return 0

                        req_exp2, req_exp3, req_exp5, req_exp7 = allowed_sum_req[current_sum_non_leading]

                        # Check if product exponents meet requirements.
                        # exp2 etc. are capped. Since CAP >= max_req_exp,
                        # `exp >= req_exp` is equivalent to `actual_exp >= req_exp`.
                        if exp2 >= req_exp2 and \
                           exp3 >= req_exp3 and \
                           exp5 >= req_exp5 and \
                           exp7 >= req_exp7:
                            return 1
                        else:
                            return 0

                state = (idx, current_sum_non_leading, exp2, exp3, exp5, exp7, tight, is_leading_zero, has_placed_zero)
                if state in memo:
                    return memo[state]

                res = 0
                upper_bound = int(num_str[idx]) if tight else 9

                for digit in range(upper_bound + 1):
                    next_tight = tight and (digit == upper_bound)

                    if is_leading_zero:
                        if digit == 0:
                            # Place a leading zero. Sum and exponents of non-leading part remain 0.
                            # is_leading_zero remains True, has_placed_zero remains False.
                            res += dp(idx + 1, 0, 0, 0, 0, 0, next_tight, True, False)
                        else:
                            # Place the first non-zero digit.
                            # Sum is this digit. Exponents are from this digit.
                            # is_leading_zero becomes False, has_placed_zero remains False.
                            d_exp2, d_exp3, d_exp5, d_exp7 = digit_exponents[digit] # digit > 0 here
                            next_exp2_capped = min(d_exp2, EXP2_CAP)
                            next_exp3_capped = min(d_exp3, EXP3_CAP)
                            next_exp5_capped = min(d_exp5, EXP5_CAP)
                            next_exp7_capped = min(d_exp7, EXP7_CAP)
                            res += dp(idx + 1, digit, next_exp2_capped, next_exp3_capped, next_exp5_capped, next_exp7_capped, next_tight, False, False)
                    else:
                        # Not in leading zero state. Place any digit.
                        # is_leading_zero remains False.
                        # has_placed_zero becomes True if digit is 0.

                        next_sum_non_leading = current_sum_non_leading + digit
                        next_has_placed_zero = has_placed_zero or (digit == 0)

                        # Update exponents of non-zero product (digit 0 contributes 0)
                        d_exp2, d_exp3, d_exp5, d_exp7 = digit_exponents[digit]
                        next_exp2 = exp2 + d_exp2
                        next_exp3 = exp3 + d_exp3
                        next_exp5 = exp5 + d_exp5
                        next_exp7 = exp7 + d_exp7

                        # Apply caps
                        next_exp2_capped = min(next_exp2, EXP2_CAP)
                        next_exp3_capped = min(next_exp3, EXP3_CAP)
                        next_exp5_capped = min(next_exp5, EXP5_CAP)
                        next_exp7_capped = min(next_exp7, EXP7_CAP)

                        res += dp(idx + 1, next_sum_non_leading, next_exp2_capped, next_exp3_capped, next_exp5_capped, next_exp7_capped, next_tight, False, next_has_placed_zero)

                memo[state] = res
                return res

            # Initial call: dp(idx, sum_non_leading, exp2, exp3, exp5, exp7, tight, is_leading_zero, has_placed_zero)
            # Start with index 0, sum of non-leading digits 0, all exponents 0, tight=True (relative to num_str), is_leading_zero=True, has_placed_zero=False.
            return dp(0, 0, 0, 0, 0, 0, True, True, False)

        # Count beautiful numbers <= r
        count_r = count_beautiful_up_to(str(r))
        # Count beautiful numbers <= l-1
        count_l_minus_1 = count_beautiful_up_to(str(l - 1))

        return count_r - count_l_minus_1