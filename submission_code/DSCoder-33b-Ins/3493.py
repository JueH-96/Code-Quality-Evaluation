class Solution:
    def maxOperations(self, s: str) -> int:
        count_ones = s.count('1')
        count_zeros = s.count('0')
        return min(count_ones, count_zeros)