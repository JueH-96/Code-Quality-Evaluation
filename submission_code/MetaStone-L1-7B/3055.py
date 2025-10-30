class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        count_ones = s.count('1')
        count_zeros = s.count('0')
        return ''.join(['1'] * (count_ones - 1) + ['0'] * count_zeros + ['1'])