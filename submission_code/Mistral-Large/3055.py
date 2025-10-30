class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        # Count the number of '1's in the string
        count_ones = s.count('1')

        # The last character must be '1' to ensure the number is odd
        # The remaining '1's should be placed as far left as possible
        # The rest of the positions should be filled with '0's

        # Create the result string with '1's followed by '0's and ending with '1'
        result = '1' * (count_ones - 1) + '0' * (len(s) - count_ones) + '1'

        return result