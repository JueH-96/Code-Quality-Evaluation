class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        # Count the number of 1s and 0s in s
        count_ones = s.count('1')
        count_zeros = s.count('0')
        
        # Since we need the result to be odd, the last digit must be '1'.
        # We place (count_ones - 1) '1's at the beginning, followed by all '0's, then the final '1'.
        
        # (count_ones - 1) ones
        leading_ones = '1' * (count_ones - 1)
        # all zeros
        zeros_part = '0' * count_zeros
        # the last digit must be '1'
        last_one = '1'
        
        return leading_ones + zeros_part + last_one