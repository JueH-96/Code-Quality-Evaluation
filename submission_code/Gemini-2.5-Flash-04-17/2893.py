from typing import List
import math

class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        n = len(nums)
        # Constraints guarantee n >= 2, so the loop range(1, n) is always valid.
        # A path must start at index 0.

        # max_ending_even: max score of a path from index 0 ending with an even number at some index i.
        # max_ending_odd: max score of a path from index 0 ending with an odd number at some index i.

        # Initialize based on the first element nums[0].
        # The only path initially is just visiting index 0.
        if nums[0] % 2 == 0:
            max_ending_even = nums[0]
            max_ending_odd = -math.inf # Represents an unreachable state initially
        else: # nums[0] % 2 != 0 (odd)
            max_ending_even = -math.inf # Represents an unreachable state initially
            max_ending_odd = nums[0]

        # Iterate from the second element (index 1) onwards.
        # At index i, we consider extending paths that ended at any index j < i.
        for i in range(1, n):
            current_val = nums[i]
            current_parity = current_val % 2

            # Calculate the maximum score achievable by extending a path to index i.
            # This requires considering the best paths ending in either parity before index i.
            if current_parity == 0: # Current number is even
                # Option 1: Extend a path ending with an even number before index i.
                # Transition even -> even costs 0. Score = score_ending_even_before + current_val.
                score_if_prev_even = max_ending_even + current_val

                # Option 2: Extend a path ending with an odd number before index i.
                # Transition odd -> even costs x. Score = score_ending_odd_before + current_val - x.
                score_if_prev_odd = max_ending_odd + current_val - x

                # The maximum score achievable by ending the path *exactly* at index i.
                potential_score_at_i = max(score_if_prev_even, score_if_prev_odd)

                # Update the overall maximum score found so far that ends with an even number.
                # This is the max of the previous best score ending even (from indices 0 to i-1)
                # and the best score ending exactly at index i (which is even).
                max_ending_even = max(max_ending_even, potential_score_at_i)

            else: # Current number is odd
                # Option 1: Extend a path ending with an even number before index i.
                # Transition even -> odd costs x. Score = score_ending_even_before + current_val - x.
                score_if_prev_even = max_ending_even + current_val - x

                # Option 2: Extend a path ending with an odd number before index i.
                # Transition odd -> odd costs 0. Score = score_ending_odd_before + current_val.
                score_if_prev_odd = max_ending_odd + current_val

                # The maximum score achievable by ending the path *exactly* at index i.
                potential_score_at_i = max(score_if_prev_even, score_if_prev_odd)

                # Update the overall maximum score found so far that ends with an odd number.
                # This is the max of the previous best score ending odd (from indices 0 to i-1)
                # and the best score ending exactly at index i (which is odd).
                max_ending_odd = max(max_ending_odd, potential_score_at_i)

        # The maximum total score is the maximum score achievable among all valid paths
        # starting at index 0 and ending at any index j (0 <= j < n).
        # This is simply the maximum of the highest score ending with an even number
        # and the highest score ending with an odd number found anywhere in the array.
        return max(max_ending_even, max_ending_odd)