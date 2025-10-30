import functools

class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7

        def calculate_count_up_to(s_val_str: str) -> int:
            N = len(s_val_str)
            total_count = 0

            # Memoization for numbers with fixed length 'length'
            # state: (idx, current_sum)
            # This cache is cleared for each new 'length' in the loop below.
            memo_fixed = {}
            def solve_fixed_len(length: int, idx: int, current_sum: int) -> int:
                # Pruning: if current_sum already exceeds max_sum, no need to continue
                if current_sum > max_sum:
                    return 0
                
                # Base case: All digits for the current 'length' are placed
                if idx == length:
                    return 1 if min_sum <= current_sum <= max_sum else 0

                state = (idx, current_sum)
                if state in memo_fixed:
                    return memo_fixed[state]

                ans = 0
                # The first digit (idx == 0) cannot be 0 for a fixed-length number.
                start_digit = 1 if idx == 0 else 0 

                for digit in range(start_digit, 10):
                    ans = (ans + solve_fixed_len(length, idx + 1, current_sum + digit)) % MOD
                
                memo_fixed[state] = ans
                return ans

            # Part 1: Count numbers with length strictly less than N (len(s_val_str))
            # These numbers are not constrained by s_val_str's digits, only by their length.
            for length in range(1, N):
                memo_fixed.clear() # Clear cache for each new length
                total_count = (total_count + solve_fixed_len(length, 0, 0)) % MOD

            # Memoization for numbers with length exactly N and value <= s_val_str
            # state: (idx, current_sum, tight)
            # This cache is specific to the current s_val_str.
            memo_N_digits = {}
            def dp_N_digits(idx: int, current_sum: int, tight: bool) -> int:
                # Pruning: if current_sum already exceeds max_sum, no need to continue
                if current_sum > max_sum:
                    return 0
                
                # Base case: All digits for length N are placed
                if idx == N:
                    return 1 if min_sum <= current_sum <= max_sum else 0
                
                state = (idx, current_sum, tight)
                if state in memo_N_digits:
                    return memo_N_digits[state]

                ans = 0
                # Determine the upper limit for the current digit based on 'tight' constraint.
                upper_bound = int(s_val_str[idx]) if tight else 9
                # The first digit (idx == 0) cannot be 0 for an N-digit number.
                start_digit = 1 if idx == 0 else 0 

                for digit in range(start_digit, upper_bound + 1):
                    # Update 'tight' flag for the next recursive call.
                    # It remains True only if 'tight' was True and current 'digit' is the 'upper_bound'.
                    new_tight = tight and (digit == upper_bound)
                    ans = (ans + dp_N_digits(idx + 1, current_sum + digit, new_tight)) % MOD
                
                memo_N_digits[state] = ans
                return ans
            
            # Part 2: Count numbers with length exactly N, and value <= s_val_str
            # We don't need to clear memo_N_digits here explicitly for each call,
            # as it's scoped to this calculate_count_up_to call.
            total_count = (total_count + dp_N_digits(0, 0, True)) % MOD

            return total_count

        # Calculate count of good numbers up to num2
        count_up_to_num2 = calculate_count_up_to(num2)

        # Calculate count of good numbers up to num1 - 1
        num1_int = int(num1)
        if num1_int == 1:
            # If num1 is "1", then numbers strictly less than num1 are none (positive integers).
            # The count function should return 0 for effectively "0".
            count_up_to_num1_minus_1 = 0 
        else:
            num1_minus_1_str = str(num1_int - 1)
            count_up_to_num1_minus_1 = calculate_count_up_to(num1_minus_1_str)
        
        # The result is count(num2) - count(num1 - 1)
        # Add MOD before modulo to handle negative results from subtraction.
        return (count_up_to_num2 - count_up_to_num1_minus_1 + MOD) % MOD