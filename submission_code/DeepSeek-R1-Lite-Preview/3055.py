class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        # Count the number of '1's and '0's
        count_ones = s.count('1')
        count_zeros = len(s) - count_ones
        
        # Reserve one '1' for the last position
        if count_ones == 0:
            # No '1's to make an odd number
            return ""
        elif count_ones == 1:
            # Only one '1', place it at the end
            return '0' * count_zeros + '1'
        else:
            # Place (count_ones - 1) '1's, then all '0's, then '1'
            return '1' * (count_ones - 1) + '0' * count_zeros + '1'