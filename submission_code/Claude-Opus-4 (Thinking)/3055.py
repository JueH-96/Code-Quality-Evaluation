class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        ones = s.count('1')
        zeros = s.count('0')
        
        # Place (ones - 1) '1's at the beginning, then all '0's, then one '1' at the end
        return '1' * (ones - 1) + '0' * zeros + '1'