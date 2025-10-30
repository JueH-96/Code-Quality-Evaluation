import sys
# Setting a higher recursion depth might be necessary for some environments,
# although the maximum depth needed here is typically small (around 10).
# sys.setrecursionlimit(2000) 

class Solution:
    """
    Solution class containing the method to count beautiful integers in a range.
    Utilizes digit dynamic programming approach.
    """
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        """
        Calculates the number of beautiful integers in the range [low, high].
        A number is beautiful if:
        1. The count of even digits equals the count of odd digits.
        2. The number is divisible by k.

        Args:
            low: The lower bound of the range (inclusive).
            high: The upper bound of the range (inclusive).
            k: The divisor for the divisibility check.

        Returns:
            The count of beautiful integers within the specified range [low, high].
        """

        # Define a fixed offset for the difference state (count_even - count_odd).
        # This offset ensures the state variable 'diff' remains non-negative.
        # MAX_DIGITS_OFFSET = 10 is chosen based on the maximum possible number of digits 
        # for high <= 10^9, which is 10. The range for diff will be [0, 20].
        MAX_DIGITS_OFFSET = 10 

        # Inner helper function that computes the count of beautiful integers up to N (inclusive).
        # This function encapsulates the DP state and logic using memoization.
        def count_up_to(N_str: str) -> int:
            """
            Counts beautiful integers in the range [1, N], where N is provided as a string.
            Uses memoization to store results of DP states.
            """
            n = len(N_str)
            # Memoization dictionary: stores computed results for DP states.
            # Key: state tuple (idx, tight, diff, rem, is_leading_zero)
            # Value: count of beautiful numbers possible from this state.
            memo = {} 

            # The core recursive function implementing the digit DP logic.
            def solve(idx: int, tight: bool, diff: int, rem: int, is_leading_zero: bool) -> int:
                """
                Recursively computes the count of valid number suffixes starting from index 'idx'.
                
                Args:
                    idx: Current digit index being processed (from left, 0-based).
                    tight: Boolean flag indicating if we are restricted by the digits of N.
                           True if the prefix matches N's prefix so far, False otherwise.
                    diff: The current difference (count_even - count_odd) + MAX_DIGITS_OFFSET. 
                          Target value is MAX_DIGITS_OFFSET for equal counts.
                    rem: The remainder of the number formed so far, modulo k. Target value is 0.
                    is_leading_zero: Boolean flag, True if we are currently placing leading zeros 
                                     or haven't placed any digits yet.

                Returns:
                    The count of beautiful numbers that can be formed from this state.
                """
                # Base case: All digits have been processed.
                if idx == n:
                    # Check if the formed number meets the conditions for being beautiful:
                    # 1. It's a positive number (not is_leading_zero indicates at least one digit was placed).
                    # 2. Equal count of even/odd digits (diff == MAX_DIGITS_OFFSET).
                    # 3. Divisible by k (rem == 0).
                    if not is_leading_zero and diff == MAX_DIGITS_OFFSET and rem == 0:
                        return 1  # Found a beautiful number
                    return 0  # Conditions not met

                # Memoization check: If state already computed, return stored result.
                state = (idx, tight, diff, rem, is_leading_zero)
                if state in memo:
                    return memo[state]

                # Calculate the upper limit for the digit at the current position 'idx'.
                # If 'tight' is True, limit is the digit N_str[idx], otherwise it's 9.
                limit = int(N_str[idx]) if tight else 9
                
                count = 0
                # Iterate through all possible digits (0 to limit) for the current position.
                for digit in range(limit + 1):
                    # Determine the 'tight' constraint for the next recursive call.
                    # It remains tight only if the current state is tight and we choose the maximum possible digit (limit).
                    new_tight = tight and (digit == limit)

                    if is_leading_zero:
                        # Handling the state where we are potentially placing leading zeros.
                        if digit == 0:
                            # If digit is 0, continue in the leading zero state.
                            # 'diff' remains neutral (MAX_DIGITS_OFFSET), 'rem' remains 0.
                            count += solve(idx + 1, new_tight, MAX_DIGITS_OFFSET, 0, True)
                        else:
                            # If digit is non-zero, this is the first actual digit of the number.
                            # Exit leading zero state (is_leading_zero becomes False for the next call).
                            # Calculate initial 'diff' based on parity of this first digit. (+1 if even, -1 if odd)
                            new_diff = MAX_DIGITS_OFFSET + (1 if digit % 2 == 0 else -1)
                            # Calculate initial remainder modulo k.
                            new_rem = digit % k
                            count += solve(idx + 1, new_tight, new_diff, new_rem, False)
                    else:
                        # Handling the state after the first non-zero digit has been placed.
                        # Update 'diff' based on the parity of the current digit.
                        new_diff = diff + (1 if digit % 2 == 0 else -1)
                        # Update remainder by appending the current digit. rem = (current_rem * 10 + digit) % k
                        new_rem = (rem * 10 + digit) % k
                        # Continue recursion with updated state, staying out of leading zero state.
                        count += solve(idx + 1, new_tight, new_diff, new_rem, False)

                # Store the computed count for the current state in the memoization table.
                memo[state] = count
                return count

            # Initiate the DP calculation by calling 'solve' for the first digit (index 0).
            # Initial state: tight=True (restricted by N_str), diff=neutral offset, rem=0, is_leading_zero=True.
            # Handle edge case where N_str represents 0. count_up_to("0") should return 0 as we count positive integers.
            if N_str == '0':
                 return 0
            return solve(0, True, MAX_DIGITS_OFFSET, 0, True)

        # Calculate the count of beautiful integers up to 'high'.
        count_high = count_up_to(str(high))
        
        # Calculate the count of beautiful integers up to 'low - 1'.
        # The function count_up_to handles str(low-1) correctly, including when low=1 resulting in str(0).
        count_low_minus_1 = count_up_to(str(low - 1))
        
        # The final result is the number of beautiful integers in the range [low, high].
        # This is obtained by subtracting the count up to low-1 from the count up to high.
        return count_high - count_low_minus_1