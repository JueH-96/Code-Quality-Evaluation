class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        count_ones = s.count('1')
        n = len(s)

        if count_ones == 0:
            return ""  # Should not happen based on constraints

        if count_ones == 1:
            return '0' * (n - 1) + '1'

        remaining_ones = count_ones - 1
        num_zeros = n - count_ones

        result = '1' * remaining_ones + '0' * num_zeros + '1'

        return result