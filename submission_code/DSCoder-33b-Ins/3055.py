class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        # Count the number of 1s in the string
        count_ones = s.count('1')

        # If there is only one '1', return the string with the '1' at the end
        if count_ones == 1:
            return '0' * (len(s) - 1) + '1'

        # If there are more than one '1', return the string with the maximum number of '1's at the start
        else:
            return '1' * count_ones + '0' * (len(s) - count_ones)