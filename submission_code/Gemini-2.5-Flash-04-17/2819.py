class Solution:
    def removeTrailingZeros(self, num: str) -> str:
        """
        Given a positive integer num represented as a string, return the
        integer num without trailing zeros as a string.

        Args:
            num: A string representing a positive integer without leading zeros.

        Returns:
            A string representing the integer num without trailing zeros.
        """
        # Python's string method rstrip(chars) removes trailing characters
        # specified in 'chars'. We want to remove trailing '0's.
        return num.rstrip('0')