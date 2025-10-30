# Global variables for DP state and memoization
memo = {}
S_DIGITS = ""
N_DIGITS = 0

# Define exponent cap indices (inclusive index for >= cap_value)
# Max required exponents for sum <= 81: exp2=6 (for 64), exp3=4 (for 81), exp5=2 (for 25, 50, 75), exp7=2 (for 49)
EXP2_CAP_IDX = 6 # States 0-6 (7 total values)
EXP3_CAP_IDX = 4 # States 0-4 (5 total values)
EXP5_CAP_IDX = 2 # States 0-2 (3 total values)
EXP7_CAP_IDX = 2 # States 0-2 (3 total values)

def get_digit_prime_exponents(digit):
    """Returns prime exponents {2, 3, 5, 7} for a non-zero digit."""
    temp = digit
    exp = {2: 0, 3: 0, 5: 0, 7: 0}
    # Assumes digit > 0
    while temp % 2 == 0:
        exp[2] += 1
        temp //= 2
    while temp % 3 == 0:
        exp[3] += 1
        temp //= 3
    while temp % 5 == 0:
        exp[5] += 1
        temp //= 5
    while temp % 7 == 0:
        exp[7] += 1
        temp //= 7
    return exp

def get_sum_prime_requirements(s):
    """
    Returns prime exponents {2, 3, 5, 7} for a sum s.
    Returns None if s has prime factors other than 2, 3, 5, 7.
    Assumes s > 0.
    """
    temp = s
    req = {2: 0, 3: 0, 5: 0, 7: 0}
    while temp > 0 and temp % 2 == 0:
        req[2] += 1
        temp //= 2
    while temp > 0 and temp % 3 == 0:
        req[3] += 1
        temp //= 3
    while temp > 0 and temp % 5 == 0:
        req[5] += 1
        temp //= 5
    while temp > 0 and temp % 7 == 0:
        req[7] += 1
        temp //= 7

    if temp > 1: # Sum has prime factors other than 2, 3, 5, 7
        return None

    return req


# state parameters: (index, tight, is_leading_zero, current_sum, exp2, exp3, exp5, exp7, saw_zero_digit)
def solve(index, tight, is_leading_zero, current_sum, exp2, exp3, exp5, exp7, saw_zero_digit):
    """
    Recursive DP function to count beautiful numbers.
    index: current digit position (from left, 0-indexed)
    tight: boolean, if restricted by the digits of the original number N
    is_leading_zero: boolean, if currently placing leading zeros
    current_sum: sum of digits placed so far (only meaningful when not is_leading_zero)
    exp2, exp3, exp5, exp7: capped exponents of prime factors in the product of non-zero digits placed so far
                           (These are only relevant if saw_zero_digit is False at the end)
    saw_zero_digit: boolean, if digit 0 has been placed at a non-leading position
    """
    # Optimized state key:
    # If is_leading_zero: (index, tight, True)
    # If not is_leading_zero: (index, tight, False, current_sum, exp2, exp3, exp5, exp7, saw_zero_digit)
    state_key = (index, tight, is_leading_zero)
    if not is_leading_zero:
        state_key = (index, tight, is_leading_zero, current_sum, exp2, exp3, exp5, exp7, saw_zero_digit)

    if state_key in memo:
        return memo[state_key]

    # Base case: all digits placed
    if index == N_DIGITS:
        if is_leading_zero:
            # Number was 0, not positive
            return 0

        # Formed a positive number. Sum is current_sum (> 0 because not is_leading_zero). Check if beautiful.
        if saw_zero_digit:
            # Contains digit 0 at a non-leading position. Product is 0. 0 % sum == 0. Beautiful.
            result = 1
        else:
            # Does not contain digit 0. Product is non-zero. Check P % S == 0.
            req = get_sum_prime_requirements(current_sum)

            if req is None:
                # Sum has prime factors other than 2, 3, 5, 7. Not beautiful.
                result = 0
            else:
                # Check if product exponents meet sum requirements
                # The state expX stores min(actual_exponent, CAP_IDX).
                # If expX == CAP_IDX, it means the actual exponent is >= CAP_IDX.
                # If expX < CAP_IDX, it means the actual exponent is exactly expX.
                # So, expX >= req[X] is the correct check.
                if exp2 >= req[2] and exp3 >= req[3] and exp5 >= req[5] and exp7 >= req[7]:
                    result = 1 # Beautiful
                else:
                    result = 0 # Not beautiful

        memo[state_key] = result
        return result

    limit_digit = int(S_DIGITS[index]) if tight else 9
    ans = 0

    # Iterate through possible digits for the current position
    for digit in range(limit_digit + 1):
        new_tight = tight and (digit == limit_digit)

        # Case 1: Still placing leading zeros
        if is_leading_zero and digit == 0:
            # Sum and product exponents remain 0 in the state representing the value built so far (which is 0).
            # saw_zero_digit remains False as this 0 is a leading zero.
            # Recurse with is_leading_zero = True and base/product state values fixed at 0/False.
            ans += solve(index + 1, new_tight, True, 0, 0, 0, 0, 0, False)
        # Case 2: Placed the first non-zero digit, or placed a digit after non-zero
        else:
            # After placing the first non-zero digit, is_leading_zero becomes False permanently.
            new_is_leading_zero = False

            # saw_zero_digit becomes True if digit 0 is used at a non-leading position.
            new_saw_zero_digit = saw_zero_digit or (digit == 0)

            new_current_sum = current_sum + digit

            # Optimization: Prune if current_sum exceeds max possible sum (81 for 9 digits)
            # Adding future non-negative digits will only increase or keep the sum the same.
            if new_current_sum > 81:
                 continue

            # Update product exponents only if the digit is non-zero.
            # If the number contains a 0 at a non-leading position (new_saw_zero_digit becomes True),
            # the actual product becomes 0, and the specific exponents (exp2..exp7) of non-zero digits
            # placed so far are no longer directly used for the final product check (0 % sum == 0).
            # However, we still need to carry them forward correctly because the 'saw_zero_digit' state
            # might be False in a recursive call path (e.g., if we later try a non-zero digit).
            # So, we always update exponents based on non-zero digits.
            new_exp2 = exp2
            new_exp3 = exp3
            new_exp5 = exp5
            new_exp7 = exp7

            if digit > 0:
                prod_exp = get_digit_prime_exponents(digit)
                # Cap the exponent at CAP_IDX
                new_exp2 = min(exp2 + prod_exp[2], EXP2_CAP_IDX)
                new_exp3 = min(exp3 + prod_exp[3], EXP3_CAP_IDX)
                new_exp5 = min(exp5 + prod_exp[5], EXP5_CAP_IDX)
                new_exp7 = min(exp7 + prod_exp[7], EXP7_CAP_IDX)

            # Recurse with is_leading_zero = False and updated state values.
            ans += solve(index + 1, new_tight, False, new_current_sum, new_exp2, new_exp3, new_exp5, new_exp7, new_saw_zero_digit)

    memo[state_key] = ans
    return ans

def count_up_to(n: int) -> int:
    """Counts beautiful numbers in the range [1, n]."""
    if n < 1:
        return 0
    global memo, S_DIGITS, N_DIGITS
    S_DIGITS = str(n)
    N_DIGITS = len(S_DIGITS)
    memo = {} # Clear memoization table for each call
    # Initial call: index=0, tight=True, is_leading_zero=True, sum=0, exp=0, saw_zero=False
    # This counts numbers from 0 up to n. Since 0 is excluded by is_leading_zero check in base case,
    # it correctly counts positive integers from 1 to n.
    return solve(0, True, True, 0, 0, 0, 0, 0, False)


class Solution:
    def beautifulNumbers(self, l: int, r: int) -> int:
        # Calculate count up to r and up to l-1, then subtract.
        count_r = count_up_to(r)
        count_l_minus_1 = count_up_to(l - 1)

        return count_r - count_l_minus_1