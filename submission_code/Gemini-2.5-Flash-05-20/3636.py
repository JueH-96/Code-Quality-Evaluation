class Solution:
    def isBalanced(self, num: str) -> bool:
        """
        Determines if a string of digits is balanced.
        A string is balanced if the sum of digits at even indices
        is equal to the sum of digits at odd indices.

        Args:
            num: A string consisting of digits only.

        Returns:
            True if the sum of digits at even indices equals the sum of digits
            at odd indices, False otherwise.
        """
        even_sum = 0
        odd_sum = 0

        # Iterate through the string with both index and digit
        for i, digit_char in enumerate(num):
            # Convert the character digit to an integer
            digit = int(digit_char)

            # Check if the current index is even or odd
            if i % 2 == 0:
                # Add to even_sum if the index is even
                even_sum += digit
            else:
                # Add to odd_sum if the index is odd
                odd_sum += digit
        
        # Return true if the sums are equal, false otherwise
        return even_sum == odd_sum