import collections # Not strictly needed for the final array-based solution, but often useful for frequency counts.
from typing import List # Needed for type hinting List[int].

class Solution:
    def countCompleteDayPairs(self, hours: List[int]) -> int:
        """
        Counts the number of pairs (i, j) such that i < j and (hours[i] + hours[j]) is a multiple of 24.

        A complete day is defined as a time duration that is an exact multiple of 24 hours.
        The condition is equivalent to (hours[i] + hours[j]) % 24 == 0.

        Args:
            hours: A list of integers representing time durations in hours.

        Returns:
            An integer denoting the number of pairs forming a complete day.

        Approach:
        We can optimize the brute-force O(n^2) approach using remainders modulo 24.
        The condition (hours[i] + hours[j]) % 24 == 0 is equivalent to
        (hours[i] % 24 + hours[j] % 24) % 24 == 0.

        Let `rem_i = hours[i] % 24` and `rem_j = hours[j] % 24`. We need `(rem_i + rem_j) % 24 == 0`.
        This means `rem_i + rem_j` must be a multiple of 24. Since `0 <= rem_i < 24` and `0 <= rem_j < 24`,
        the sum `rem_i + rem_j` can be either 0 or 24.
        - If `rem_i = 0`, we need `rem_j = 0`.
        - If `rem_i > 0`, we need `rem_j = 24 - rem_i`.

        We can combine these: the required remainder `rem_j` for a given `rem_i` is `(24 - rem_i) % 24`.

        We can iterate through the `hours` list once, maintaining a count of the remainders encountered so far.
        For each `hour` at index `i`, let its remainder be `current_remainder = hour % 24`.
        We need to find how many previous elements (at index `j < i`) have the `target_remainder = (24 - current_remainder) % 24`.
        We use a frequency map (or an array of size 24) `remainders_count` to store the counts of remainders seen so far.

        Algorithm:
        1. Initialize `count = 0`.
        2. Initialize `remainders_count = [0] * 24`.
        3. Iterate through each `hour` in the `hours` list:
           a. Calculate `current_remainder = hour % 24`.
           b. Calculate `target_remainder = (24 - current_remainder) % 24`.
           c. Add `remainders_count[target_remainder]` to `count`. This represents the number of pairs found where the current `hour` is the second element of the pair.
           d. Increment `remainders_count[current_remainder]` to include the current `hour`'s remainder for future pairings.
        4. Return `count`.

        Time complexity: O(n), where n is the length of the `hours` list, as we iterate through the list once.
        Space complexity: O(1), as the `remainders_count` array has a fixed size of 24, independent of the input size.
        """
        # Initialize a frequency array of size 24 to store counts of remainders modulo 24.
        # remainders_count[k] will store the number of elements `h` seen so far such that `h % 24 == k`.
        remainders_count = [0] * 24
        
        # Initialize the total count of pairs that form a complete day.
        count = 0

        # Iterate through each hour value in the input list.
        for hour in hours:
            # Calculate the remainder of the current hour when divided by 24.
            current_remainder = hour % 24

            # Calculate the complementary remainder needed for the sum with the current hour
            # to be a multiple of 24. If the current remainder is `r`, the target remainder
            # is `(24 - r) % 24`.
            target_remainder = (24 - current_remainder) % 24

            # Add the count of previously seen numbers with the `target_remainder` to the total count.
            # Each such previously seen number forms a complete day pair with the current number.
            # Since we process elements from left to right, this ensures we count pairs (j, i) where j < i.
            count += remainders_count[target_remainder]

            # Increment the frequency count for the `current_remainder`.
            # This makes the current number available for pairing with subsequent numbers in the list.
            remainders_count[current_remainder] += 1

        # Return the final calculated count of complete day pairs.
        return count