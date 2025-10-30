class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        # Count the number of '1's and '0's in the string
        count1 = s.count('1')
        count0 = len(s) - count1
        
        # We must place one '1' at the end to make the number odd
        count1 -= 1
        
        # For the maximum value, put all remaining '1's first, then all '0's, then the final '1'
        return '1' * count1 + '0' * count0 + '1'