class Solution:
    def removeTrailingZeros(self, num: str) -> str:
        """
        Removes all trailing zeros from a string representation of a positive integer.

        The problem specifies that 'num' is a positive integer represented as a string,
        consists only of digits, and doesn't have any leading zeros.

        This can be efficiently solved using Python's string method `rstrip()`.
        The `rstrip()` method returns a copy of the string with trailing characters
        removed. If an argument is provided to `rstrip()`, it removes all occurrences
        of characters present in the argument from the end of the string.

        In this case, we want to remove only the character '0' from the end of the string.

        Args:
            num: A string representing a positive integer.

        Returns:
            A string representing the integer num without any trailing zeros.
        """
        # Use the rstrip() method to remove all trailing '0' characters.
        # Example:
        # "51230100".rstrip('0') -> "512301"
        # "123".rstrip('0') -> "123"
        # "1000".rstrip('0') -> "1"
        return num.rstrip('0')