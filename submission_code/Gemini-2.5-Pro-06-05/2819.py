class Solution:
    def removeTrailingZeros(self, num: str) -> str:
        """
        Removes trailing zeros from a string representation of a positive integer.

        Args:
            num: A string representing a positive integer.

        Returns:
            The integer num without trailing zeros as a string.
        """
        # The rstrip() method removes any trailing characters from the end of a string.
        # We specify '0' as the character to be removed. It will continue
        # removing '0's from the right until it encounters a different
        # character or reaches the beginning of the string.
        return num.rstrip('0')