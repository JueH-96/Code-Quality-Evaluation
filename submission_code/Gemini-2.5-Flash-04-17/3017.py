class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:

        # Memoization dictionary and shared variables for the DP function
        # These need to be accessible and modifiable by the nested dp function.
        memo = {}
        k_val = 0
        s_str = ""
        n = 0

        # The recursive DP function
        # idx: current digit position (0 to n)
        # tight: boolean, true if constrained by the digits of s_str
        # is_leading_zeros: boolean, true if the number formed so far is 0
        # even_cnt: count of even digits in the non-leading part formed so far
        # odd_cnt: count of odd digits in the non-leading part formed so far
        # rem: value of the non-leading part formed so far modulo k
        def dp(idx, tight, is_leading_zeros, even_cnt, odd_cnt, rem):
            # Use the nonlocal keyword to modify variables in the nearest enclosing scope
            nonlocal memo, k_val, s_str, n

            # Check memoization
            memo_key = (idx, tight, is_leading_zeros, even_cnt, odd_cnt, rem)
            if memo_key in memo:
                return memo[memo_key]

            # Base case: Reached the end of the string length.
            # We have successfully placed digits for all n positions (0 to n-1).
            # The number formed is composed of the non-leading digits placed.
            if idx == n:
                # If we were still in the leading zeros state, the number formed is 0.
                # The problem constraints 0 < low <= high ensure 0 is not in the range, so we don't count 0.
                if is_leading_zeros:
                    return 0
                
                # If not in leading zeros state, we formed a non-zero number.
                # This number is beautiful if the count of its even digits equals the count of its odd digits
                # AND it is divisible by k.
                # The state variables even_cnt, odd_cnt, and rem reflect these properties for the
                # number formed by the non-leading digits.
                # The total number of non-leading digits is even_cnt + odd_cnt. This is the length L.
                # The condition even_cnt == odd_cnt implies L = even_cnt + even_cnt = 2 * even_cnt,
                # which means the length L must be even. This is necessary for even_cnt == odd_cnt.
                if even_cnt == odd_cnt and rem == 0:
                    return 1 # Found a beautiful number
                else:
                    return 0 # Not beautiful

            # Recursive step: Iterate through possible digits to place at the current index.
            upper_bound = int(s_str[idx]) if tight else 9
            count = 0

            for digit in range(upper_bound + 1):
                
                if is_leading_zeros and digit == 0:
                    # If we are placing leading zeros and the current digit is 0:
                    # - We remain in the leading zero state.
                    # - The counts (even, odd) and remainder do not change (they are effectively 0 for number 0).
                    # - The tightness constraint propagates: still tight if the upper_bound was 0.
                    count += dp(idx + 1, tight and (digit == upper_bound), True, even_cnt, odd_cnt, rem)
                else:
                    # If we are placing the first non-zero digit (transition from is_leading_zeros=True)
                    # OR we were already past leading zeros (is_leading_zeros=False):
                    # - We are now definitely NOT in the leading zeros state (new_is_leading_zeros = False).
                    # - Update the counts and remainder based on the current digit being a non-leading digit.
                    new_even_cnt = even_cnt + (1 if digit % 2 == 0 else 0)
                    new_odd_cnt = odd_cnt + (1 if digit % 2 != 0 else 0)
                    new_rem = (rem * 10 + digit) % k_val
                    new_is_leading_zeros = False # We just placed a non-leading digit

                    # The tightness constraint propagates: still tight if the current digit is the upper_bound.
                    count += dp(idx + 1, tight and (digit == upper_bound), False, new_even_cnt, new_odd_cnt, new_rem)

            # Store and return the result for the current state.
            memo[memo_key] = count
            return count

        # Helper function to count beautiful numbers up to a given integer (inclusive).
        def count_beautiful_up_to(num_int, k_val_arg):
            # Reset memoization for each upper bound number calculation.
            # Update the shared variables used by the dp function.
            nonlocal memo, k_val, s_str, n
            memo = {}
            k_val = k_val_arg
            s_str = str(num_int)
            n = len(s_str)
            
            # Start the DP from index 0.
            # Initially tight (restricted by the digits of num_str).
            # Initially in the leading zeros state (no non-zero digits placed yet).
            # Initial counts (even, odd) and remainder are 0.
            # We call dp(0, tight=True, is_leading_zeros=True, even_cnt=0, odd_cnt=0, rem=0).
            # The DP will count all beautiful numbers X such that 0 <= X <= num_int.
            # Since 0 is not in the range [low, high], we don't need to explicitly subtract 0.
            # The base case handles 0 correctly (returns 0 if is_leading_zeros is True at the end).
            return dp(0, True, True, 0, 0, 0)

        # The number of beautiful integers in [low, high] is
        # (count of beautiful integers <= high) - (count of beautiful integers <= low - 1).
        # Need to handle low - 1 = 0 case. count_beautiful_up_to(0, k) should be 0.
        # Our DP handles num_int = 0 correctly: s_str becomes "0", n=1.
        # dp(0, T, T, 0, 0, 0) -> digit=0 -> dp(1, T, T, 0, 0, 0) -> Base case idx=1, is_lz=T -> returns 0. Correct.
        
        result_high = count_beautiful_up_to(high, k)
        result_low_minus_1 = count_beautiful_up_to(low - 1, k)

        return result_high - result_low_minus_1