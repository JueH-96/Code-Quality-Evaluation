class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        # Count total number of '1's and '0's in s
        ones = s.count('1')
        zeros = len(s) - ones
        
        # We must place exactly one '1' at the end to make the number odd
        # The remaining ones should all go to the front to maximize the prefix,
        # followed by all zeros, and then the final '1'.
        rem_ones = ones - 1
        
        return '1' * rem_ones + '0' * zeros + '1'