class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        n = len(s)
        count_ones = s.count('1')
        
        if count_ones == 1:
            return '0' * (n - 1) + '1'
        elif count_ones > 1:
            ones_to_place = count_ones - 1
            zeros_to_place = n - count_ones
            return '1' * ones_to_place + '0' * zeros_to_place + '1'
        else:
            # According to constraints, there is at least one '1', so this won't be reached
            return ''