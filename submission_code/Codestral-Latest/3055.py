class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        # Count the number of '1's in the string
        count_ones = s.count('1')

        # The maximum odd binary number will have all '1's at the beginning followed by '0's
        # The last bit must be '1' to ensure the number is odd
        result = '1' * (count_ones - 1) + '0' * (len(s) - count_ones) + '1'

        return result