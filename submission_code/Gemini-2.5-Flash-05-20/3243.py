class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        
        def count_powerful_up_to(num: int, limit: int, s: str) -> int:
            num_str = str(num)
            len_num = len(num_str)
            len_s = len(s)

            # If the suffix 's' is longer than 'num' itself,
            # no number <= num can have 's' as a suffix.
            if len_s > len_num:
                return 0
            
            memo = {} # Memoization table for DP states

            # dp(idx, tight, is_started)
            # idx: Current digit position we are filling (from left, 0 to len_num - 1)
            # tight: True if we are restricted by digits of num_str, False otherwise.
            #        e.g., if num_str = "123", and we are at idx=0, tight is true, so we can only place 0 or 1.
            # is_started: True if we have already placed at least one non-zero digit.
            #             This handles leading zeros and ensures we count positive integers.
            def dp(idx: int, tight: bool, is_started: bool) -> int:
                # Base case: All digits considered.
                if idx == len_num:
                    # If is_started is True, we successfully formed a positive number.
                    # This number has length len_num and was built respecting 's' suffix.
                    return 1 if is_started else 0

                state = (idx, tight, is_started)
                if state in memo:
                    return memo[state]

                ans = 0
                # Determine the upper bound for the current digit based on 'tight' constraint.
                upper_bound_digit = int(num_str[idx]) if tight else 9

                # Determine the position where the suffix 's' should start within the num_str.
                # E.g., if num_str="6000" (len=4), s="124" (len=3), suffix starts at idx = 4-3 = 1.
                # So prefix is at idx=0. Suffix is at idx=1,2,3.
                suffix_start_idx = len_num - len_s

                # Scenario 1: We are filling prefix digits (before the suffix 's' part)
                if idx < suffix_start_idx:
                    # Option A: Place a leading zero. Only possible if 'is_started' is False.
                    # This path effectively shortens the number's length.
                    if not is_started:
                        ans += dp(idx + 1, tight and (0 == upper_bound_digit), False)
                    
                    # Option B: Place a non-zero digit (to start the number or continue building it).
                    # 'start_digit' is 1 if we haven't placed any non-zero digit yet, otherwise 0.
                    start_digit = 1 if not is_started else 0
                    
                    # Iterate through possible digits for the current position, respecting 'limit' and 'upper_bound_digit'.
                    for digit in range(start_digit, min(upper_bound_digit, limit) + 1):
                        ans += dp(idx + 1, tight and (digit == upper_bound_digit), True) # is_started becomes True
                
                # Scenario 2: We are filling suffix digits (from 's')
                else: 
                    # Calculate the corresponding digit from 's'.
                    s_digit_val = int(s[idx - suffix_start_idx])
                    
                    # Check if the forced s_digit_val satisfies current constraints.
                    # 1. It must be <= 'upper_bound_digit' (due to 'tight' constraint).
                    # 2. It must be <= 'limit' (problem constraints guarantee s's digits are <= limit).
                    if s_digit_val <= upper_bound_digit:
                        # Update 'tight' flag for the next recursive call.
                        new_tight = tight and (s_digit_val == upper_bound_digit)
                        
                        # Update 'is_started' flag.
                        # If s_digit_val is > 0, then the number is definitely started.
                        # Since 's' does not have leading zeros, s[0] is never '0'.
                        # This means if idx == suffix_start_idx and is_started was False,
                        # new_is_started will become True because s_digit_val (s[0]) will be > 0.
                        new_is_started = is_started or (s_digit_val > 0)
                        
                        ans += dp(idx + 1, new_tight, new_is_started)

                memo[state] = ans
                return ans

        # Calculate the count of powerful integers up to 'finish'.
        count_finish = count_powerful_up_to(finish, limit, s)
        # Calculate the count of powerful integers up to 'start - 1'.
        count_start_minus_1 = count_powerful_up_to(start - 1, limit, s)
        
        # The total count in range [start, finish] is the difference.
        return count_finish - count_start_minus_1