import math

class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        # This set will store unique multisets of digits that can form a k-palindromic integer.
        # A multiset is represented as a tuple of 10 counts (c0, c1, ..., c9).
        found_good_multisets = set()
        
        # Precompute factorials for efficiency in count_permutations
        MAX_N = 10 # n is at most 10
        factorials = [1] * (MAX_N + 1)
        for i in range(2, MAX_N + 1):
            factorials[i] = factorials[i-1] * i

        def count_permutations(counts_tuple, num_digits):
            """
            Calculates the number of distinct integers that can be formed
            from a given multiset of digits, ensuring no leading zeros.
            counts_tuple: tuple of digit counts (c0, c1, ..., c9)
            num_digits: total number of digits (n)
            """
            # Calculate total permutations assuming no leading zero constraint
            denom = 1
            for count in counts_tuple:
                denom *= factorials[count]
            
            total_perms = factorials[num_digits] // denom
            
            # Subtract permutations with leading zeros if '0' is present in the multiset
            if counts_tuple[0] > 0:
                # If a '0' is at the first position, we are forming an (n-1)-digit number
                # from the remaining (n-1) digits, where one '0' is used.
                remaining_denom = factorials[counts_tuple[0] - 1] # count of 0s decreases by 1
                for i in range(1, 10): # counts of other digits remain same
                    remaining_denom *= factorials[counts_tuple[i]]
                
                perms_with_leading_zero = factorials[num_digits - 1] // remaining_denom
                return total_perms - perms_with_leading_zero
            else:
                return total_perms

        # half_len represents the number of digits in the first half of the palindrome.
        # For n=1, half_len=0.
        # For n=2, half_len=1.
        # For n=3, half_len=1.
        half_len = n // 2

        def dfs(idx, current_digits_first_half):
            """
            Recursively builds the first half of a potential k-palindromic number.
            
            idx: The current position (from 0 to half_len - 1) in current_digits_first_half
                 that we are trying to fill.
            current_digits_first_half: A list of digits chosen so far for the first half.
            """
            if idx == half_len:
                # Base case: The first half of the palindrome (of length half_len) is complete.
                s_left_half = "".join(map(str, current_digits_first_half))
                s_right_half = s_left_half[::-1] # Reverse the first half for the second half
                
                if n % 2 == 0:
                    # n is even: The k-palindromic number B is first_half + reversed(first_half).
                    b_str = s_left_half + s_right_half
                    
                    # For n > 1, current_digits_first_half[0] is never '0' due to 'start_digit' logic.
                    # So b_str[0] will never be '0' if n is even and > 0.
                    # The n=0 case is not considered as problem states n>=1.
                    
                    # Calculate the full multiset of digits for B
                    full_digit_counts = [0] * 10
                    for digit in current_digits_first_half:
                        full_digit_counts[digit] += 2 # Each digit in the first half appears twice in B
                    
                    # Convert to integer and check divisibility by k
                    b_val = int(b_str)
                    if b_val % k == 0:
                        found_good_multisets.add(tuple(full_digit_counts))
                else:
                    # n is odd: The k-palindromic number B is first_half + middle_digit + reversed(first_half).
                    # We iterate through all possible digits for the middle position.
                    for d_mid in range(10):
                        b_str = s_left_half + str(d_mid) + s_right_half
                        
                        # Handle leading zeros for B.
                        # If n=1, s_left_half is "", so b_str is just str(d_mid). If d_mid=0, B is "0", which is invalid.
                        # For n > 1, s_left_half[0] is guaranteed not to be '0' by the 'start_digit' logic below.
                        # Thus, b_str[0] will not be '0' for n > 1.
                        if b_str[0] == '0': # This condition is only relevant for n=1 and d_mid=0.
                            continue # Skip invalid numbers like "0" for n=1.

                        # Calculate the full multiset of digits for B
                        full_digit_counts = [0] * 10
                        for digit in current_digits_first_half:
                            full_digit_counts[digit] += 2 # Digits in first half appear twice
                        full_digit_counts[d_mid] += 1 # The middle digit appears once
                        
                        # Convert to integer and check divisibility by k
                        b_val = int(b_str)
                        if b_val % k == 0:
                            found_good_multisets.add(tuple(full_digit_counts))
                return

            # Recursive step: Choose the next digit for the first half
            # The first digit of any n-digit number (idx == 0) cannot be '0'.
            # For subsequent digits (idx > 0), '0' is allowed.
            start_digit = 1 if idx == 0 else 0
            
            for d in range(start_digit, 10):
                dfs(idx + 1, current_digits_first_half + [d])

        # Start the DFS process to find all k-palindromic multisets
        dfs(0, [])
        
        # Sum up the counts of good integers from all unique multisets found
        total_good_integers = 0
        for counts_tuple in found_good_multisets:
            total_good_integers += count_permutations(counts_tuple, n)
            
        return total_good_integers