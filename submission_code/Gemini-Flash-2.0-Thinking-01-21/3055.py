class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        ones_count = s.count('1')
        zeros_count = len(s) - ones_count
        if ones_count == 0:
            return ""
        if ones_count == 1:
            return '0' * zeros_count + '1'
        else:
            return '1' * (ones_count - 1) + '0' * zeros_count + '1'