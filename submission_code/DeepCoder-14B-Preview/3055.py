class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        ones = s.count('1')
        n = len(s)
        remaining_ones = ones - 1
        first_part = '1' * remaining_ones + '0' * (n - 1 - remaining_ones)
        return first_part + '1'