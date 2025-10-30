class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        count_ones = s.count('1')
        zero_count = len(s) - count_ones
        return '1' * (count_ones - 1) + '0' * zero_count + '1'