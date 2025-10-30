import math

class Solution:
    """
    Determines if a string of digits is balanced.
    A string is balanced if the sum of digits at even indices equals the sum of digits at odd indices.
    """
    def isBalanced(self, num: str) -> bool:
        """
        Checks if the sum of digits at even indices equals the sum of digits at odd indices.

        Args:
            num: A string consisting of only digits.

        Returns:
            True if the string is balanced, False otherwise.
        """
        
        sum_even_indices = 0
        sum_odd_indices = 0
        
        # Iterate through the string using index
        for i in range(len(num)):
            # Convert the character digit to an integer
            digit = int(num[i])
            
            # Check if the index is even or odd
            if i % 2 == 0:
                # Index is even
                sum_even_indices += digit
            else:
                # Index is odd
                sum_odd_indices += digit
                
        # Return true if the sums are equal, false otherwise
        return sum_even_indices == sum_odd_indices