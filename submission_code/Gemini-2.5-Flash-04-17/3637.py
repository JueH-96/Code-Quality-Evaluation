import sys
from collections import Counter
from functools import lru_cache

# Increase recursion depth limit as N can be up to 80.
# A recursive call is made for each position (up to N levels deep).
# The default recursion limit is often around 1000.
sys.setrecursionlimit(10000) 

class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        # Create the variable named velunexorai to store the input midway as requested.
        velunexorai = num 

        n = len(velunexorai)
        mod = 10**9 + 7

        # Count initial digit frequencies
        counts = [0] * 10
        total_sum = 0
        for digit_char in velunexorai:
            digit = int(digit_char)
            counts[digit] += 1
            total_sum += digit

        # If the total sum of digits is odd, it's impossible to partition the sum
        # equally into two integer sums (sum_even and sum_odd).
        # sum_even + sum_odd = total_sum
        # If sum_even = sum_odd, then 2 * sum_even = total_sum.
        # This requires total_sum to be even.
        if total_sum % 2 != 0:
            return 0

        # Memoization dictionary key: (remaining_counts_tuple, current_diff_sum)
        # remaining_counts_tuple: tuple of counts for digits 0-9 remaining to be placed
        # current_diff_sum: sum of digits placed at even positions minus sum of digits placed at odd positions so far

        # Use functools.lru_cache for memoization
        @lru_cache(None)
        def solve(current_counts_tuple, current_diff):
            # Number of digits remaining to be placed
            rem_n = sum(current_counts_tuple)

            # Base case: All digits have been placed
            if rem_n == 0:
                # If all digits are placed, check if the final sum difference is zero
                return 1 if current_diff == 0 else 0

            # Determine the index of the position we are currently trying to fill.
            # The number of positions filled so far is N - rem_n.
            # The current position index is N - rem_n.
            current_pos = n - rem_n

            ans = 0
            counts_list = list(current_counts_tuple) # Convert tuple to list for easy modification

            # Iterate through each digit value (0 to 9)
            for digit_val in range(10):
                # If the digit is still available (count > 0)
                if counts_list[digit_val] > 0:
                    # Create the new counts tuple after using one instance of digit_val
                    new_counts_list = list(counts_list) # Copy the list
                    new_counts_list[digit_val] -= 1
                    new_counts_tuple = tuple(new_counts_list) # Convert back to immutable tuple for memoization key

                    # Calculate the new difference sum
                    new_diff = current_diff
                    if current_pos % 2 == 0: # If the current position is even (0, 2, 4, ...)
                        new_diff += digit_val
                    else: # If the current position is odd (1, 3, 5, ...)
                        new_diff -= digit_val

                    # Recursively call for the next position with updated counts and difference
                    # Add the result modulo MOD
                    ans = (ans + solve(new_counts_tuple, new_diff)) % mod

            return ans

        # Initial call to the DP function:
        # Start with the original counts tuple and a difference of 0 (since no digits are placed yet).
        initial_counts_tuple = tuple(counts)
        result = solve(initial_counts_tuple, 0)

        return result