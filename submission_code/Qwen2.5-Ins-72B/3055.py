class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        # Count the number of '1's in the string
        count_ones = s.count('1')
        
        # We need at least one '1' at the end to make the number odd
        # The rest of the '1's should be at the beginning to maximize the number
        # The remaining positions will be filled with '0's
        return '1' * (count_ones - 1) + '0' * (len(s) - count_ones) + '1'