class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        count_ones = s.count('1')
        count_zeros = s.count('0')
        
        # The maximum odd binary number is formed by placing all but one '1' at the front,
        # then all the zeros, and finally one '1' at the end to ensure the number is odd.
        return '1' * (count_ones - 1) + '0' * count_zeros + '1'