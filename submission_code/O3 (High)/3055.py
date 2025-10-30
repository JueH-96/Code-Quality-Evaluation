class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        """
        The maximum odd binary number is obtained by:
        1. Placing exactly one '1' as the least-significant bit (to keep the number odd).
        2. Putting all remaining '1's as far to the left as possible.
        3. Filling the remaining positions in the middle with '0's.
        
        If the input contains k ones and z zeros (k ≥ 1), the layout becomes:
            '1' * (k - 1)  +  '0' * z  +  '1'
        """
        ones  = s.count('1')
        zeros = len(s) - ones
        # (ones - 1) leading '1's, then all '0's, finally one '1'
        return '1' * (ones - 1) + '0' * zeros + '1'