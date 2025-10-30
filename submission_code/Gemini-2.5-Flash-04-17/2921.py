import sys
# sys.setrecursionlimit(2000) # Default recursion depth is usually sufficient for depth ~100

MOD = 10**9 + 7

class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:

        def count_le_with_zero(s: str) -> int:
            """
            Counts the number of stepping numbers X such that 0 <= X <= int(s).
            A number X is considered a stepping number if its standard decimal
            representation satisfies the stepping property. The number 0 is included.
            """
            n = len(s)
            # memo[index][tight][prev_digit_code][is_leading_zero]
            # index: current digit position (0 to n)
            # tight: boolean, whether restricted by the upper bound string s
            # prev_digit_code: 0-9 for the value of the previous significant digit,
            #                  10 for the state where no significant digit has been placed yet (is_leading_zero is True)
            # is_leading_zero: boolean, whether we are currently placing leading zeros
            memo = [[[[(-1) for _ in range(2)] for _ in range(11)] for _ in range(2)] for _ in range(n + 1)]

            # index: current digit position we are filling (0 to n-1)
            # tight: boolean, True if current prefix matches s's prefix, restricting current digit's upper bound
            # prev_digit_val: the value of the digit placed at index - 1. Only relevant if not in leading zero state. Use 10 as sentinel for leading zero state.
            # is_leading_zero: boolean, True if all digits from 0 to index - 1 were 0.
            def dp(index: int, tight: bool, prev_digit_val: int, is_leading_zero: bool) -> int:
                if index == n:
                    # We have successfully formed a sequence of n digits.
                    # This sequence represents a number <= int(s) that satisfies the stepping
                    # condition on its significant digits. Count this number.
                    return 1

                # Determine the memoization key for the previous digit state.
                # If we are in the leading zero state, the specific value of prev_digit_val (which is 10)
                # doesn't matter for the stepping rule, only the fact that it's leading.
                # We use 10 as the index to represent the leading zero state for prev_digit_code.
                pd_code_idx = prev_digit_val if not is_leading_zero else 10

                if memo[index][tight][pd_code_idx][is_leading_zero] != -1:
                    return memo[index][tight][pd_code_idx][is_leading_zero]

                res = 0
                # The maximum digit we can place at the current position
                upper_bound = int(s[index]) if tight else 9

                if is_leading_zero:
                    # Option 1: Place 0 at the current position.
                    # The number remains in the leading zero state at the next position.
                    # The previous digit value remains the sentinel (10) for the next state's logic.
                    # We can always place 0 if the upper_bound allows it (0 <= upper_bound, which is always true).
                    # The next state is tight only if the current digit (0) equals the upper_bound digit from s.
                    res = (res + dp(index + 1, tight and (0 == upper_bound), 10, True)) % MOD

                    # Option 2: Place a non-zero digit d at the current position.
                    # This is the first significant digit. The number is no longer in the leading zero state.
                    # d can range from 1 up to the upper_bound.
                    for d in range(1, upper_bound + 1):
                        # The previous digit for the next state is this digit d.
                        # The next state is tight only if the current digit (d) equals the upper_bound digit from s.
                        # is_leading_zero becomes False.
                        res = (res + dp(index + 1, tight and (d == upper_bound), d, False)) % MOD

                else: # Not in the leading zero state, we have placed at least one significant digit.
                    # The previous significant digit is prev_digit_val (0-9).
                    prev_digit = prev_digit_val

                    # The current digit d must satisfy abs(d - prev_digit) == 1.
                    # Possible values for d are prev_digit - 1 and prev_digit + 1.
                    possible_digits = []
                    if prev_digit > 0:
                        possible_digits.append(prev_digit - 1)
                    if prev_digit < 9:
                        possible_digits.append(prev_digit + 1)

                    for d in possible_digits:
                        # The current digit d must be within the allowed range [0, upper_bound].
                        # The lower bound is 0 since we are not placing the very first digit of the number.
                        if d <= upper_bound:
                            # The previous digit for the next state is this digit d.
                            # The next state is tight only if the current digit (d) equals the upper_bound digit from s.
                            # is_leading_zero remains False.
                            res = (res + dp(index + 1, tight and (d == upper_bound), d, False)) % MOD

                # Store and return the result for the current state
                memo[index][tight][pd_code_idx][is_leading_zero] = res
                return res

            # The initial call starts at index 0, is restricted by the input string s (tight=True),
            # is in the leading zero state initially (is_leading_zero=True),
            # and the prev_digit_val is the sentinel (10) as there is no previous significant digit yet.
            # This counts stepping numbers in [0, int(s)], including 0.
            return dp(0, True, 10, True)

        # Helper function to subtract 1 from a number represented as a string.
        # Handles borrowing and resulting leading zeros.
        # subtract_one_str("100") -> "99"
        # subtract_one_str("1") -> "0"
        def subtract_one_str(s: str) -> str:
            n = len(s)
            digits = [int(c) for c in s]
            i = n - 1
            while i >= 0:
                if digits[i] > 0:
                    digits[i] -= 1
                    break
                else: # digits[i] == 0, need to borrow from the left
                    digits[i] = 9 # Change 0 to 9
                    i -= 1

            # After the loop, if i < 0, it means we borrowed from a leading 1 (e.g., "100" became [0, 9, 9]).
            # We need to construct the result string, stripping leading zeros.

            result_digits = []
            leading_zero = True
            for digit in digits:
                if leading_zero and digit == 0:
                    # Skip leading zeros, unless the number is just 0
                    continue
                else:
                    # Once a non-zero digit is found, the leading zero phase is over
                    leading_zero = False
                    result_digits.append(str(digit))

            # If result_digits is empty, it means the original number was "1" and became [0] after decrementing.
            # The number 1 - 1 is 0, represented as "0".
            if not result_digits:
                return "0"

            # Join the digits to form the result string
            return "".join(result_digits)

        # The count of stepping numbers in the inclusive range [low, high]
        # is equal to (count of stepping numbers X <= high) - (count of stepping numbers X < low).
        # Count stepping numbers X < low is the same as count stepping numbers X <= (low - 1).
        # Our helper function `count_le_with_zero(S)` counts stepping numbers in [0, int(S)], including 0.
        # Since the problem constraints state `1 <= int(low)`, the number 0 is never included in the range [low, high].
        # Therefore, the count of stepping numbers in [low, high] is:
        # (Count of stepping numbers in [0, int(high)]) - (Count of stepping numbers in [0, int(low) - 1]).
        # Both `count_le_with_zero(high)` and `count_le_with_zero(subtract_one_str(low))` will correctly handle
        # the count of 0 (if present) such that the subtraction yields the count only within the desired range [low, high].

        count_high = count_le_with_zero(high)
        count_low_minus_1 = count_le_with_zero(subtract_one_str(low))

        # The result must be non-negative before taking modulo.
        ans = (count_high - count_low_minus_1 + MOD) % MOD

        return ans