class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        # Count the number of '1's in the input string
        ones = s.count('1')
        # Compute the number of '0's in the input string
        zeros = len(s) - ones
        
        # We need an odd binary number, so we place one '1' at the end.
        # The remaining '1's (ones - 1) should be at the beginning
        # to maximize the binary value, and the zeros go in the middle.
        return '1' * (ones - 1) + '0' * zeros + '1'