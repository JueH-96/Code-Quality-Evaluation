class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        count_ones = s.count('1')
        count_zeros = len(s) - count_ones
        
        if count_ones == 1:
            return '0' * count_zeros + '1'
        else:
            return '1' * (count_ones - 1) + '0' * count_zeros + '1'