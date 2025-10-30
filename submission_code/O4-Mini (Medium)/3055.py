class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        # Count the total number of '1's and '0's in the input
        ones = s.count('1')
        zeros = len(s) - ones
        
        # One '1' must be at the end to make the number odd
        # The remaining (ones - 1) '1's should be placed in front,
        # followed by all the '0's, to maximize the binary value.
        if ones > 0:
            leading_ones = ones - 1
            return '1' * leading_ones + '0' * zeros + '1'
        else:
            # (By problem statement, there is at least one '1', so we never actually hit this branch.)
            return '0' * zeros