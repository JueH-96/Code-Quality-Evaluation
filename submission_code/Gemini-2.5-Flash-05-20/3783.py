from typing import List

class Solution:
    def permute(self, n: int, k: int) -> List[int]:
        # Precompute factorials for calculating permutations of remaining numbers
        fact = [1] * (n + 1)
        for i in range(2, n + 1):
            fact[i] = fact[i-1] * i

        # Calculate initial number of odd and even integers available from 1 to n
        total_odd_count = (n + 1) // 2
        total_even_count = n // 2

        # dp[o][e][next_is_odd] stores the number of alternating *patterns*
        # that can be formed using 'o' odd numbers and 'e' even numbers,
        # where the first number in this sub-pattern must be odd (if next_is_odd=1)
        # or even (if next_is_odd=0).
        # This table effectively stores 0 or 1, as there's at most one valid pattern
        # for a given (o, e, starting_parity) combination.
        dp = [[[0] * 2 for _ in range(total_even_count + 1)] for _ in range(total_odd_count + 1)]

        # Base cases: an empty sequence (0 odd, 0 even) is a valid pattern.
        # dp[0][0][0] means 0 odds, 0 evens, and the *next* number needed was even.
        # dp[0][0][1] means 0 odds, 0 evens, and the *next* number needed was odd.
        # Both are possible ways to "complete" a pattern.
        dp[0][0][0] = 1 
        dp[0][0][1] = 1
        
        # Fill DP table for all possible (o, e) combinations
        for o in range(total_odd_count + 1):
            for e in range(total_even_count + 1):
                if o == 0 and e == 0:
                    continue # Already handled by base cases
                
                # To form a pattern where the first number must be odd (1):
                # We place an odd number, then the remaining sub-pattern must start with an even.
                if o > 0:
                    dp[o][e][1] = dp[o-1][e][0]
                
                # To form a pattern where the first number must be even (0):
                # We place an even number, then the remaining sub-pattern must start with an odd.
                if e > 0:
                    dp[o][e][0] = dp[o][e-1][1]

        # Array to keep track of used numbers
        used_nums = [False] * (n + 1)
        
        # Result permutation list
        result = []
        
        # Track remaining counts of odd and even numbers as we build the permutation
        current_odd_count = total_odd_count
        current_even_count = total_even_count
        
        # Track parity of the previously placed number (-1 for no previous element)
        prev_parity = -1 

        # Build the permutation element by element (N positions)
        for _ in range(n):
            found_digit_for_current_pos = False
            # Iterate through possible numbers in lexicographical order (1 to n)
            for num in range(1, n + 1):
                if used_nums[num]:
                    continue # Skip already used numbers

                current_num_parity = num % 2

                # Check if placing 'num' maintains the alternating parity rule
                is_valid_next_parity = False
                if prev_parity == -1: # First element, any parity is valid
                    is_valid_next_parity = True
                elif prev_parity != current_num_parity: # Parity must be different from previous
                    is_valid_next_parity = True

                if not is_valid_next_parity:
                    continue # 'num' cannot be placed here due to parity rules

                # Calculate remaining counts if 'num' is chosen
                o_rem_after_choice = current_odd_count - (1 if current_num_parity == 1 else 0)
                e_rem_after_choice = current_even_count - (1 if current_num_parity == 0 else 0)

                # Determine the parity the *next* element in the sequence must have.
                # This is used to query the DP table.
                next_expected_parity_for_dp = 1 if current_num_parity == 0 else 0 # 1 means next must be odd, 0 means next must be even

                num_permutations_if_chosen = 0
                if o_rem_after_choice >= 0 and e_rem_after_choice >= 0:
                    # Get the count of valid parity patterns for the remaining numbers
                    pattern_count = dp[o_rem_after_choice][e_rem_after_choice][next_expected_parity_for_dp]
                    
                    # Total permutations = (pattern count) * (ways to arrange remaining odd nums) * (ways to arrange remaining even nums)
                    num_permutations_if_chosen = pattern_count * fact[o_rem_after_choice] * fact[e_rem_after_choice]

                # Check if 'k' falls within the range of permutations starting with 'num'
                if k <= num_permutations_if_chosen:
                    # 'num' is the correct digit for the current position
                    result.append(num)
                    used_nums[num] = True
                    if current_num_parity == 1:
                        current_odd_count -= 1
                    else:
                        current_even_count -= 1
                    prev_parity = current_num_parity
                    found_digit_for_current_pos = True
                    break # Move to the next position in the permutation
                else:
                    # 'num' is not the one, subtract its permutations from 'k' and try the next number
                    k -= num_permutations_if_chosen
            
            if not found_digit_for_current_pos:
                # If no digit could be placed at the current position, 'k' is out of range
                return []
        
        return result