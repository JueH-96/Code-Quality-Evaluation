class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7

        # Helper function using digit DP to count numbers x in [0, int(num_str)]
        # such that min_sum <= digit_sum(x) <= max_sum.
        def solve(num_str: str) -> int:
            n = len(num_str)
            # Memoization table: memo[index][current_sum][tight][is_leading_zero]
            # `current_sum` stores the sum of non-leading digits.
            # The state `current_sum` ranges from 0 up to `max_sum`.
            memo = {}

            # dp(index, current_sum, tight, is_leading_zero)
            # index: current digit position we are considering (from 0 to n-1)
            # current_sum: sum of non-leading digits placed so far (from index 0 to index-1)
            # tight: boolean, True if we are restricted by num_str's digits at current/future positions
            # is_leading_zero: boolean, True if all digits placed so far are 0
            def dp(index: int, current_sum: int, tight: bool, is_leading_zero: bool) -> int:
                # Base case: All digits processed.
                if index == n:
                    # We have formed a number. Check if its digit sum is valid.
                    # `current_sum` correctly holds the sum of non-leading digits (which is the digit sum of the number).
                    return 1 if min_sum <= current_sum <= max_sum else 0

                # Memoization check
                state = (index, current_sum, tight, is_leading_zero)
                if state in memo:
                    return memo[state]

                ans = 0
                # Determine the upper bound for the current digit
                upper_bound = int(num_str[index]) if tight else 9

                # Iterate through possible digits for the current position
                for digit in range(upper_bound + 1):
                    # Calculate new state parameters

                    # Calculate new sum and leading zero state
                    if is_leading_zero and digit == 0:
                        # If currently placing leading zeros and place 0:
                        # sum remains 0, still placing leading zeros.
                        new_current_sum = current_sum # which must be 0 if is_leading_zero is True
                        new_is_leading_zero = True
                    else:
                        # If not placing leading zeros, or place non-zero digit:
                        # sum increases by digit, no longer placing leading zeros.
                        new_current_sum = current_sum + digit
                        new_is_leading_zero = False

                    # Pruning: If the potential new sum exceeds max_sum, this path is invalid.
                    # This check ensures we never recurse into states where current_sum > max_sum.
                    if new_current_sum > max_sum:
                        continue

                    # Calculate new tight state: remains tight only if it was tight and we chose the upper_bound digit.
                    new_tight = tight and (digit == upper_bound)

                    # Recursive call
                    ans = (ans + dp(index + 1, new_current_sum, new_tight, new_is_leading_zero)) % MOD

                # Store result in memoization table
                memo[state] = ans
                return ans

            # Start the DP from the first digit (index 0), initial sum 0, tight=True (restricted by num_str),
            # and is_leading_zero=True (initially, we can place leading zeros, which handles numbers
            # shorter than num_str and the number 0 itself).
            # This function counts numbers x in [0, int(num_str)] with valid digit sum.
            return dp(0, 0, True, True)

        # Count good numbers in the range [num1, num2]
        # This is equivalent to (count good numbers <= num2) - (count good numbers <= num1 - 1)

        # Count good numbers x such that 0 <= x <= int(num2) and min_sum <= digit_sum(x) <= max_sum
        count_le_num2 = solve(num2)

        # Count good numbers x such that 0 <= x <= int(num1) - 1 and min_sum <= digit_sum(x) <= max_sum
        # Convert num1 to int, subtract 1, convert back to string.
        # Since 1 <= num1, num1-1 >= 0. str(0) is handled correctly by solve.
        num1_int = int(num1)
        num1_minus_1_str = str(num1_int - 1)
        count_le_num1_minus_1 = solve(num1_minus_1_str)

        # The result is the difference, modulo MOD. Ensure result is non-negative.
        result = (count_le_num2 - count_le_num1_minus_1 + MOD) % MOD

        return result