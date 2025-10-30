import sys

class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        
        # It's possible that the recursion depth limit is an issue for very specific cases,
        # though for N up to 10^15, P_max_str length is at most ~15, which is well within typical Python limits.
        # If concerned, sys.setrecursionlimit can be used, but usually not needed for these constraints.

        s_val = int(s)
        len_s = len(s)
        power_of_10_len_s = 10**len_s

        # Defines count_powerful_le(N_boundary)
        # This function counts powerful integers x such that 1 <= x <= N_boundary.
        # Powerful integers x are of the form P * (10^len_s) + s_val.
        # We need to count P >= 0 such that P * (10^len_s) + s_val <= N_boundary,
        # and all digits of P are <= limit.
        def count_le_N(N_boundary: int) -> int:
            # If N_boundary is less than s_val, no powerful number (which must be >= s_val)
            # can be <= N_boundary.
            if N_boundary < s_val:
                return 0
            
            # Calculate the maximum possible value for the prefix P.
            # P_max = floor((N_boundary - s_val) / (10^len_s))
            P_max = (N_boundary - s_val) // power_of_10_len_s
            
            # Convert P_max to string to iterate over its digits in DP.
            P_max_str = str(P_max)
            
            # Memoization dictionary for the digit DP.
            # Fresh for each call to count_le_N (i.e., for `finish` and `start-1`).
            memo = {} 

            # Digit DP function to count P in [0, P_max] with all digits <= limit.
            # idx: current digit position in P_max_str being considered.
            # tight: boolean, true if P built so far matches prefix of P_max_str.
            # is_leading: boolean, true if P built so far is all zeros (e.g., "0", "00").
            def dfs_count_P(idx: int, tight: bool, is_leading: bool) -> int:
                # Base case: If all digits of P_max_str have been processed.
                if idx == len(P_max_str):
                    return 1 # One valid P has been formed.
                
                state = (idx, tight, is_leading)
                if state in memo:
                    return memo[state]

                ans = 0
                # Determine the upper bound for the current digit choice.
                # If tight, current digit cannot exceed P_max_str[idx]. Otherwise, it can be up to 9.
                upper_digit_choice = int(P_max_str[idx]) if tight else 9
                
                for digit in range(upper_digit_choice + 1):
                    # Constraint: all digits of P must be <= limit.
                    # Since limit >= 1, digit 0 is always <= limit.
                    # If digit > 0, it must satisfy digit <= limit.
                    # This is concisely written as:
                    if digit > limit:
                        continue
                    
                    # Update tight constraint for the next recursive call.
                    # It remains tight only if current tight is true AND we pick the max possible digit.
                    new_tight = tight and (digit == upper_digit_choice)
                    
                    # Update is_leading for the next recursive call.
                    # It remains true if current is_leading is true AND we pick 0.
                    new_is_leading = is_leading and (digit == 0)
                    
                    ans += dfs_count_P(idx + 1, new_tight, new_is_leading)
                
                memo[state] = ans
                return ans

            # Initial call to the digit DP.
            # Start at index 0, with tight constraint active, and P is initially "leading zeros".
            return dfs_count_P(0, True, True)

        # The total count is count_le_N(finish) - count_le_N(start - 1).
        ans_finish = count_le_N(finish)
        ans_start_minus_1 = count_le_N(start - 1)
        
        return ans_finish - ans_start_minus_1