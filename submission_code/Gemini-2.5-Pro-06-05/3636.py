class Solution:
    def isBalanced(self, num: str) -> bool:
        """
        Checks if a string of digits is balanced.

        A string is balanced if the sum of the digits at even indices is equal
        to the sum of the digits at odd indices.

        Args:
            num: A string consisting of only digits.

        Returns:
            True if the number is balanced, False otherwise.
        """

        # Calculate the sum of digits at even indices (0, 2, 4, ...).
        # We use string slicing with a step of 2 (num[::2]) to get all
        # characters at even indices. Then, we use a generator expression
        # to convert each character to an integer and sum them up.
        even_sum = sum(int(digit) for digit in num[::2])
        
        # Calculate the sum of digits at odd indices (1, 3, 5, ...).
        # Slicing from the first index with a step of 2 (num[1::2]) gets
        # all characters at odd indices.
        odd_sum = sum(int(digit) for digit in num[1::2])
        
        # The number is balanced if the two sums are equal.
        return even_sum == odd_sum