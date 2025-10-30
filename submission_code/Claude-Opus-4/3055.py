class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        # Count the number of 1s and 0s
        ones = s.count('1')
        zeros = s.count('0')
        
        # We need at least one 1 at the end to make it odd
        # Place remaining 1s at the beginning to maximize the value
        # Place all 0s in between
        
        # Build the result string
        result = '1' * (ones - 1) + '0' * zeros + '1'
        
        return result