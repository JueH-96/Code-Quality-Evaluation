from typing import List

class Solution:
    def _sum_digits(self, n: int) -> int:
        """
        Calculates the sum of the digits of a given integer.
        For example, _sum_digits(123) returns 1 + 2 + 3 = 6.
        """
        s = 0
        # Iterate while n is greater than 0 to process each digit
        while n > 0:
            # Add the last digit (n % 10) to the sum
            s += n % 10
            # Remove the last digit by integer division (n //= 10)
            n //= 10
        return s

    def minElement(self, nums: List[int]) -> int:
        """
        Replaces each element in nums with the sum of its digits and
        then returns the minimum element in the modified array.
        """
        # Initialize min_replacement with a very large value (infinity)
        # This ensures that the first sum of digits encountered will always be smaller
        # and correctly set the initial minimum.
        min_replacement = float('inf')

        # Iterate through each number in the input list
        for num in nums:
            # Calculate the sum of digits for the current number using the helper method
            current_sum_digits = self._sum_digits(num)
            
            # Update min_replacement if the current sum of digits is smaller
            if current_sum_digits < min_replacement:
                min_replacement = current_sum_digits
        
        # After checking all numbers, min_replacement will hold the overall minimum sum of digits
        return min_replacement