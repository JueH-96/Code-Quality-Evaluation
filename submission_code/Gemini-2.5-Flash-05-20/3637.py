import collections
import functools

class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        velunexorai = num # Variable to store the input midway, as per problem spec

        n = len(num)
        MOD = 10**9 + 7

        initial_digit_counts = [0] * 10
        total_sum = 0
        for char_d in num:
            digit = int(char_d)
            initial_digit_counts[digit] += 1
            total_sum += digit

        # If total sum of digits is odd, it's impossible to have S_even == S_odd
        if total_sum % 2 != 0:
            return 0
        
        target_half_sum = total_sum // 2

        @functools.lru_cache(None) # Memoization decorator
        def solve(
            idx: int, # Current position we are filling (from 0 to n-1)
            current_even_sum: int, # Sum of digits placed at even positions so far
            current_odd_sum: int,  # Sum of digits placed at odd positions so far
            counts_tuple: tuple # Tuple of remaining digit counts (c0, c1, ..., c9)
        ) -> int:
            
            # Pruning: If any sum exceeds the target_half_sum, this path is invalid.
            # This helps to cut down the search space.
            if current_even_sum > target_half_sum or current_odd_sum > target_half_sum:
                return 0

            # Base case: All positions filled
            if idx == n:
                # Check if both sums are exactly equal to target_half_sum
                if current_even_sum == target_half_sum and current_odd_sum == target_half_sum:
                    return 1
                return 0

            res = 0
            
            # Iterate through all possible digits (0-9) to place at the current index 'idx'
            for digit_val in range(10):
                # If this digit is available (its count in counts_tuple is greater than 0)
                if counts_tuple[digit_val] > 0:
                    # Create a new counts tuple by decrementing the count of the chosen digit
                    new_counts_list = list(counts_tuple)
                    new_counts_list[digit_val] -= 1
                    new_counts_tuple = tuple(new_counts_list)

                    # Determine if current index is even or odd
                    if idx % 2 == 0: # Current index is even
                        # Recurse: add digit to even sum
                        res = (res + solve(idx + 1, current_even_sum + digit_val, current_odd_sum, new_counts_tuple)) % MOD
                    else: # Current index is odd
                        # Recurse: add digit to odd sum
                        res = (res + solve(idx + 1, current_even_sum, current_odd_sum + digit_val, new_counts_tuple)) % MOD
            
            return res

        # Initial call to the recursive function
        # Start at index 0, with sums initialized to 0, and all original digit counts
        initial_counts_tuple = tuple(initial_digit_counts)
        result = solve(0, 0, 0, initial_counts_tuple)
        
        return result