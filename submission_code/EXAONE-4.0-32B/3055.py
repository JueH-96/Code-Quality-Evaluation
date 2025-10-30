class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        ones_count = s.count('1')
        n = len(s)
        return '1' * (ones_count - 1) + '0' * (n - ones_count) + '1'