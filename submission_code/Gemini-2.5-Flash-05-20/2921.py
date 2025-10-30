import functools

class Solution:
    MOD = 10**9 + 7

    def countSteppingNumbers(self, low: str, high: str) -> int:
        
        # Helper function to subtract 1 from a string representation of a number.
        # This is necessary because numbers can be very large (up to 100 digits).
        def minus_one(s_num_str: str) -> str:
            s_list = list(s_num_str)
            n = len(s_list)
            idx = n - 1
            
            # Find the rightmost non-zero digit and decrement it.
            # If a digit is '0', it becomes '9' and we continue borrowing from the left.
            while idx >= 0:
                if s_list[idx] == '0':
                    s_list[idx] = '9'
                    idx -= 1
                else:
                    s_list[idx] = str(int(s_list[idx]) - 1)
                    break
            
            # Handle cases where subtraction leads to a shorter number or "0".
            # Example: "100" -> "99", "10" -> "9", "1" -> "0".
            if s_list[0] == '0':
                if n == 1: # Original number was "1", result is "0"
                    return "0"
                else: # Original number started with "1" (e.g., "10", "100"), now starts with "0" (e.g., "09", "099")
                    # Remove the leading '0'.
                    return "".join(s_list[1:])
            else:
                # No leading '0' formed or number was already single digit non-zero.
                return "".join(s_list)

        # Digit DP function to count stepping numbers up to s_num_str (inclusive).
        # This function effectively counts stepping numbers in the range [1, int(s_num_str)].
        def count(s_num_str: str) -> int:
            N = len(s_num_str)

            # @functools.lru_cache(None) is used for memoization.
            # Since `dp` is a nested function, a new `dp` function object (and thus a new cache)
            # is created for each call to the outer `count` function, which is suitable here.
            @functools.lru_cache(None)
            def dp(idx: int, prev_digit: int, is_less: bool, is_started: bool) -> int:
                """
                Args:
                    idx: Current digit position we are filling (from 0 to N-1).
                    prev_digit: The digit placed at `idx-1`. Used for stepping condition.
                                If `is_started` is False, this is a sentinel value (e.g., 10)
                                as no relevant previous digit exists.
                    is_less: True if the number formed so far is already strictly less than
                             the prefix of `s_num_str`. If True, subsequent digits can be 0-9.
                    is_started: True if we have placed at least one non-zero digit.
                                False means we are currently forming a sequence of leading zeros
                                or haven't placed any digit yet.
                Returns:
                    The count of valid stepping numbers that can be formed from `idx` onwards
                    given the current state.
                """
                if idx == N:
                    # If we have reached the end of the string, a number is formed.
                    # It's a valid stepping number only if it contains at least one non-zero digit.
                    return 1 if is_started else 0 

                res = 0
                # Determine the upper limit for the current digit.
                # If `is_less` is True, we are no longer constrained by `s_num_str`'s digits, so max is 9.
                # Otherwise, the max digit is `s_num_str[idx]`.
                upper_bound = int(s_num_str[idx]) if not is_less else 9

                for digit in range(upper_bound + 1):
                    if not is_started:
                        # Case 1: We are still handling leading zeros or trying to start the number.
                        if digit == 0:
                            # Continue with leading zeros. `prev_digit` remains sentinel (10).
                            # `is_less` is updated if placing 0 makes the current prefix strictly less than `s_num_str`'s prefix.
                            res = (res + dp(idx + 1, 10, is_less or (digit < upper_bound), False)) % Solution.MOD
                        else:
                            # This is the first non-zero digit, so the number has truly started.
                            # `prev_digit` is now `digit`.
                            res = (res + dp(idx + 1, digit, is_less or (digit < upper_bound), True)) % Solution.MOD
                    else:
                        # Case 2: The number has already started (at least one non-zero digit placed).
                        # Apply the stepping number condition: adjacent digits must differ by exactly 1.
                        if abs(digit - prev_digit) == 1:
                            res = (res + dp(idx + 1, digit, is_less or (digit < upper_bound), True)) % Solution.MOD
                
                return res

            # Initial call to the DP function:
            # - Start at index 0.
            # - `prev_digit` is 10 (sentinel, indicating no previous digit for comparison).
            # - `is_less` is False (initially, we are restricted by `s_num_str`).
            # - `is_started` is False (no non-zero digits placed yet).
            return dp(0, 10, False, False)

        # Calculate the count of stepping numbers up to 'high'.
        ans_high = count(high)
        
        # Calculate 'low - 1' as a string.
        low_minus_1_str = minus_one(low)
        
        # Calculate the count of stepping numbers up to 'low - 1'.
        ans_low_minus_1 = count(low_minus_1_str)

        # The final result is (count(high) - count(low-1)) modulo MOD.
        # Add MOD before taking modulo to ensure the result is positive even if subtraction yields a negative number.
        return (ans_high - ans_low_minus_1 + Solution.MOD) % Solution.MOD