class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        # Count 1s and 0s
        ones = s.count('1')
        zeros = s.count('0')
        
        # We need at least one '1' at the end to make it odd
        # Put remaining '1's at the beginning, then all '0's, then one '1' at the end
        result = '1' * (ones - 1) + '0' * zeros + '1'
        
        return result