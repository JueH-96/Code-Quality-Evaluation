class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        count_ones = s.count('1')
        count_zeros = s.count('0')
        # We place (count_ones - 1) ones at the front, all zeros in the middle,
        # and one '1' at the end to ensure the number is odd.
        return '1' * (count_ones - 1) + '0' * count_zeros + '1'