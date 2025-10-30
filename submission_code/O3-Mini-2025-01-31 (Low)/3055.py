class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        # Count the number of ones and zeros in the string
        ones = s.count('1')
        zeros = s.count('0')
        # Ensure that the final number is odd, meaning the last digit must be '1'
        # Reserve one '1' for the last position
        return '1' * (ones - 1) + '0' * zeros + '1'