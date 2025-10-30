class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        count_ones = s.count('1')
        count_zeros = len(s) - count_ones
        # To make the number odd, we need at least one '1' at the end
        # The rest '1's should be placed at the beginning to maximize the number
        # So, the arrangement is: (count_ones - 1) '1's, then count_zeros '0's, then one '1'
        result = '1' * (count_ones - 1) + '0' * count_zeros + '1'
        return result