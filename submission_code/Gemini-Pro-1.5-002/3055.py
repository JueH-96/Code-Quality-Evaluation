class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        count_ones = s.count('1')
        result = ""
        for _ in range(count_ones - 1):
            result += '1'
        for _ in range(len(s) - count_ones):
            result += '0'
        result += '1'
        return result