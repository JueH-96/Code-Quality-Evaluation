import collections
from typing import List

class Solution:
    def totalNumbers(self, digits: List[int]) -> int:
        """
        Counts the number of distinct three-digit even numbers that can be formed
        using the given digits, respecting the frequency of each digit.

        Args:
            digits: A list of digits (0-9).

        Returns:
            The number of distinct three-digit even numbers.
        """
        # Count the frequency of each digit in the input digits list.
        # Using collections.Counter is efficient for this.
        digits_counts = collections.Counter(digits)
        
        # Create an empty set to store the distinct valid numbers found.
        # A set automatically handles uniqueness, ensuring we count distinct numbers.
        valid_numbers = set()
        
        # Iterate through all possible three-digit numbers (100 to 999).
        # A three-digit number ranges from 100 to 999.
        # Iterating this range automatically handles the constraint that the first digit cannot be 0.
        for num in range(100, 1000):
            # Check if the current number is even.
            if num % 2 == 0:
                # Extract the digits of the current three-digit number.
                d1 = num // 100         # Hundreds digit
                d2 = (num // 10) % 10   # Tens digit
                d3 = num % 10           # Units digit
                
                # Count the frequency of the digits required to form the current number.
                num_counts = collections.Counter([d1, d2, d3])
                
                # Check if the required digits (and their counts) for the current number
                # are available in the input digits list.
                # The Counter comparison (<=) checks if the count of each digit needed
                # is less than or equal to the count available in the input.
                if num_counts <= digits_counts:
                    # If the number can be formed using the available digits, add it to the set.
                    valid_numbers.add(num)
        
        # The size of the set valid_numbers is the total count of distinct
        # three-digit even numbers that can be formed.
        return len(valid_numbers)