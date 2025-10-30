from typing import List
from collections import Counter # Import Counter for frequency counting

class Solution:
    def totalNumbers(self, digits: List[int]) -> int:
        """
        Calculates the number of distinct three-digit even numbers that can be formed
        using the digits from the input list `digits`.

        Args:
            digits: A list of single digits (0-9). The length is between 3 and 10.
                    Each digit can be used at most as many times as it appears in this list
                    for constructing a single number.

        Returns:
            An integer representing the count of distinct three-digit even numbers
            that satisfy the conditions:
            1. The number must be three digits long (100-999).
            2. The number must be even.
            3. The number must be formed using only digits from the `digits` list,
               respecting the frequency of each digit available.
            4. The number cannot have leading zeros (this is implicitly handled by
               considering numbers in the range 100-999).
        """
        # Count the frequency of each digit available in the input list.
        # Example: digits = [0, 2, 2] -> available_counts = {0: 1, 2: 2}
        available_counts = Counter(digits)

        # Initialize a set to store the unique valid three-digit even numbers found.
        # Using a set automatically handles distinctness.
        distinct_numbers = set()

        # Iterate through all possible three-digit numbers (100 to 999).
        # We only need to check even numbers, so we start at 100 and step by 2.
        # The loop range is [100, 1000), so it covers 100, 102, ..., 998.
        for num in range(100, 1000, 2):

            # Extract the hundreds, tens, and units digits of the current even number 'num'.
            # Example: num = 124 -> d1 = 1, d2 = 2, d3 = 4
            d1 = num // 100        # Hundreds digit
            d2 = (num % 100) // 10 # Tens digit
            d3 = num % 10          # Units digit (guaranteed even by the loop step)

            # Count the frequency of digits required to form the number 'num'.
            # Example: num = 202 -> required_counts = Counter({2: 2, 0: 1})
            required_counts = Counter([d1, d2, d3])

            # Flag to track if the current number 'num' can be formed using the available digits.
            can_form = True

            # Check if we have enough of each required digit in our available digits.
            # Iterate through the digits needed for 'num' and their required counts.
            for digit, required_count in required_counts.items():
                # Compare the required count with the available count for that digit.
                # available_counts[digit] will return 0 if the digit is not in the input list,
                # which correctly handles cases where a needed digit is simply not available.
                if available_counts[digit] < required_count:
                    # If the available count is less than required for any digit,
                    # this number cannot be formed.
                    can_form = False
                    # No need to check other digits for this number, break the inner loop.
                    break

            # If the inner loop completed without setting can_form to False,
            # it means all required digits are available in sufficient quantities.
            if can_form:
                # Add the valid formed number to our set.
                distinct_numbers.add(num)

        # The final result is the number of distinct valid numbers found.
        return len(distinct_numbers)