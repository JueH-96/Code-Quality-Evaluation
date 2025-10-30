from typing import List

class Solution:
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        # This approach uses a frequency map to count the occurrences of each remainder
        # when hours are divided by 24.
        # It iterates through the hours array once, achieving O(n) time complexity.
        # The space complexity is O(1) because the size of the remainder map is fixed at 24.

        # Create a list to store counts of remainders 0 through 23.
        # Index i stores the count of hours h such that h % 24 == i.
        remainder_counts = [0] * 24

        # Initialize the counter for complete day pairs.
        pair_count = 0

        # Iterate through each hour in the input list.
        for hour in hours:
            # Calculate the remainder of the current hour when divided by 24.
            current_remainder = hour % 24

            # To form a complete day (sum is a multiple of 24) with the current hour,
            # a previous hour must have a remainder such that (previous_remainder + current_remainder) % 24 == 0.
            # Let previous_remainder be R. We need (R + current_remainder) % 24 == 0.
            # This implies R + current_remainder must be a multiple of 24.
            # Since 0 <= R <= 23 and 0 <= current_remainder <= 23, the sum R + current_remainder is between 0 and 46.
            # Thus, R + current_remainder must be either 0 or 24.

            # We need R = (24 - current_remainder) % 24.
            # If current_remainder is 0, (24 - 0) % 24 = 24 % 24 = 0.
            # If current_remainder is 5, (24 - 5) % 24 = 19 % 24 = 19.
            # If current_remainder is 23, (24 - 23) % 24 = 1 % 24 = 1.
            # This formula correctly identifies the remainder needed from a previous element.
            required_remainder_for_complement = (24 - current_remainder) % 24

            # Before updating the count for the current remainder, check how many previous hours
            # have the `required_remainder_for_complement`. Each of these previous hours forms a
            # complete day pair with the current hour.
            # We add this count to our total `pair_count`.
            # This step correctly counts pairs (j, i) where j < i, as we are iterating from i=0 to n-1
            # and looking for complements among elements already processed (indices < i).
            pair_count += remainder_counts[required_remainder_for_complement]

            # After processing the current hour and counting pairs it forms with previous hours,
            # increment the count for the current hour's remainder. This makes the current hour
            # available to form pairs with subsequent hours in the list.
            remainder_counts[current_remainder] += 1

        # Return the total number of complete day pairs found.
        return pair_count