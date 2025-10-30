import sys

# Increase recursion depth for potential edge cases if needed, though unlikely for this problem
# sys.setrecursionlimit(2000) 

class Solution:
    """
    Given a positive integer num represented as a string, return the integer num 
    without trailing zeros as a string.
    """
    def removeTrailingZeros(self, num: str) -> str:
        """
        Removes trailing zeros from a string representation of a positive integer.

        Args:
            num: A string representing a positive integer.

        Returns:
            The string representation of the integer with trailing zeros removed.
        """

        # Method 1: Using rstrip()
        # The rstrip() method removes specified characters from the end (right side)
        # of a string. By default, it removes whitespace, but we can specify '0'.
        return num.rstrip('0')

        # # Method 2: Iterating from the end (alternative implementation)
        # n = len(num)
        # # Find the index of the last non-zero character
        # last_non_zero_index = n - 1
        # while last_non_zero_index >= 0 and num[last_non_zero_index] == '0':
        #     last_non_zero_index -= 1
        
        # # If all characters were zeros (which shouldn't happen based on constraints,
        # # but defensively), the index would be -1. The problem states num is positive,
        # # so there will always be at least one non-zero digit.
        
        # # Slice the string up to and including the last non-zero character
        # return num[:last_non_zero_index + 1]