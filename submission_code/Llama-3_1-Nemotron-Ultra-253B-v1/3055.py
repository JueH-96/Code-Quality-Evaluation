class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        count_ones = s.count('1')
        remaining_ones = count_ones - 1
        remaining_zeros = len(s) - count_ones
        max_part = '1' * remaining_ones + '0' * remaining_zeros
        return max_part + '1'