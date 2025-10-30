class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        # Count the number of '1's and '0's
        ones = s.count('1')
        zeros = s.count('0')
        
        # If there is only one '1', it must be at the end to form an odd number
        if ones == 1:
            return '0' * (zeros - 1) + '01'
        
        # Otherwise, place all '1's at the beginning followed by all '0's except one,
        # which should be at the end to make the number odd
        return '1' * (ones - 1) + '0' * zeros + '1'