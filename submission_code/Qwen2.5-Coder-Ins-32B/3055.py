class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        count_ones = s.count('1')
        if count_ones == 1:
            return '0' * (len(s) - 1) + '1'
        return '1' * (count_ones - 1) + '0' * (len(s) - count_ones) + '1'