class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        # Count the number of ones and zeros in the string.
        ones = s.count('1')
        zeros = s.count('0')
        # To form an odd number, the last digit must be '1'.
        # Place one '1' at the end and build the most significant part 
        # by arranging the remaining ones (ones-1) first (for maximum value)
        # followed by all the zeros.
        return "1" * (ones - 1) + "0" * zeros + "1"