class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        """
        We want to count how many integers x satisfy:
           num1 <= x <= num2
           and min_sum <= digit_sum(x) <= max_sum.
        Since the range can be very large (up to 10^22),
        we use a digit-DP approach.

        Steps:
        1) Define a helper function count_up_to(num_str) that returns
           how many integers x in [0, num_str] have a digit sum in [min_sum, max_sum].
        2) Use digit-DP to compute this count.
        3) The final result is count_up_to(num2) - count_up_to(num1 - 1).
        """

        MOD = 10**9 + 7
        
        # Digit DP to count valid numbers in [0 .. num_str].
        def count_up_to(num_str: str) -> int:
            # Edge case: if the number is '0', handle separately
            if num_str == "0":
                # No numbers in [0,0] if we need sum in [min_sum, max_sum] and min_sum >= 1
                # If min_sum = 0, we'd need special handling, but constraints say min_sum >= 1
                return 0

            digits = list(map(int, num_str))
            n = len(digits)

            # dp(pos, sum_so_far, restricted, leading) returns the count of valid ways
            # to form numbers from position 'pos' onward with a digit sum = sum_so_far,
            # where 'restricted' indicates if we are bound by the prefix of num_str,
            # and 'leading' indicates if we've only chosen leading zeros so far.
            from functools import lru_cache

            @lru_cache(None)
            def dp(pos, sum_so_far, restricted, leading):
                # Base case: if we've placed all digits
                if pos == n:
                    # Check if the final sum of digits is within [min_sum, max_sum].
                    return 1 if (min_sum <= sum_so_far <= max_sum) else 0
                
                result = 0
                limit = digits[pos] if restricted else 9
                
                for dig in range(limit + 1):
                    new_sum = sum_so_far
                    new_leading = leading
                    
                    if leading and dig == 0:
                        # sum_so_far stays the same if it's still leading zeros
                        pass
                    else:
                        # Once we place a non-zero digit, leading = False
                        if leading and dig != 0:
                            new_leading = False
                        new_sum += dig
                    
                    # If we've exceeded max_sum, break early
                    if new_sum > max_sum:
                        break
                    
                    new_restricted = restricted and (dig == limit)
                    result = (result + dp(pos + 1, new_sum, new_restricted, new_leading)) % MOD
                
                return result
            
            return dp(0, 0, True, True)
        
        # Convert num1 to int, subtract 1, handle if it goes below '0'
        def prev_num_str(s: str) -> str:
            # returns the string for (int(s) - 1)
            # guaranteed s >= "1"
            x = int(s) - 1
            return str(x)
        
        # Count how many numbers are valid up to num2
        count_num2 = count_up_to(num2)
        
        # Count how many numbers are valid up to num1-1
        # If num1 is '1', num1 - 1 = '0' is a valid call
        # because digit-DP for "0" was handled in the function with an edge case.
        if num1 == "1":
            count_num1_minus_1 = 0
        else:
            count_num1_minus_1 = count_up_to(prev_num_str(num1))
        
        return (count_num2 - count_num1_minus_1) % MOD