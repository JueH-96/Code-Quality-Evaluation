import math

class Solution:
    def sumOfTheDigitsOfHarshadNumber(self, x: int) -> int:
        """
        Calculates the sum of the digits of an integer x.
        Checks if x is divisible by the sum of its digits (Harshad number).
        Returns the sum of digits if it's a Harshad number, otherwise returns -1.

        Args:
            x: The input integer (1 <= x <= 100).

        Returns:
            The sum of the digits of x if x is a Harshad number, otherwise -1.
        """
        
        # Calculate the sum of the digits of x
        digit_sum = 0
        temp_x = x  # Use a temporary variable to extract digits without modifying original x
        
        # Iterate through the digits using modulo and integer division
        while temp_x > 0:
            digit = temp_x % 10  # Get the last digit
            digit_sum += digit   # Add it to the sum
            temp_x //= 10      # Remove the last digit
            
        # Alternatively, using string conversion (equally valid for small constraints):
        # digit_sum = sum(int(digit) for digit in str(x))

        # Check if x is divisible by the sum of its digits
        if x % digit_sum == 0:
            # If divisible, it's a Harshad number, return the sum of digits
            return digit_sum
        else:
            # If not divisible, it's not a Harshad number, return -1
            return -1