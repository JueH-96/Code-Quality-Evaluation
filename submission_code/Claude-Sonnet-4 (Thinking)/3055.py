class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        # Count the number of '1's
        count_ones = s.count('1')
        
        # Total length of the string
        n = len(s)
        
        # We need at least one '1' at the end to make it odd
        # So we have (count_ones - 1) '1's to place at the beginning
        ones_at_beginning = count_ones - 1
        
        # The number of zeros is total length minus total ones
        zeros = n - count_ones
        
        # Construct the result: ones at beginning + zeros in middle + one '1' at end
        result = '1' * ones_at_beginning + '0' * zeros + '1'
        
        return result