from typing import List
from collections import Counter

class Solution:
    def totalNumbers(self, digits: List[int]) -> int:
        """
        Determines the number of distinct three-digit even numbers that can be formed
        using the given digits.

        Args:
            digits: A list of digits.

        Returns:
            The count of distinct three-digit even numbers.
        """
        # Count the occurrences of each digit available in the input list
        available_counts = Counter(digits)

        # Set to store the distinct valid numbers found
        valid_numbers = set()

        # Iterate through all possible three-digit even numbers
        # A three-digit number ranges from 100 to 999.
        # An even number must end in 0, 2, 4, 6, or 8.
        # So, we can iterate from 100 to 998, stepping by 2.
        for num in range(100, 1000, 2):
            # For the current number, count the occurrences of each digit required
            required_counts = Counter(int(d) for d in str(num))

            # Check if the current number can be formed using the available digits
            can_form = True
            for digit, count in required_counts.items():
                # If the number of required digits is more than available,
                # this number cannot be formed.
                if available_counts.get(digit, 0) < count:
                    can_form = False
                    break # No need to check further for this number

            # If the number can be formed, add it to our set of valid numbers
            if can_form:
                valid_numbers.add(num)

        # The size of the set is the count of distinct valid numbers
        return len(valid_numbers)