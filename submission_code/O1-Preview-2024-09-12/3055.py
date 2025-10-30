class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        num_ones = s.count('1')
        num_zeros = len(s) - num_ones
        result = '1' * (num_ones - 1) + '0' * num_zeros + '1'
        return result