class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        # Count the number of '1's in the string
        ones_count = s.count('1')
        
        # Compute the number of '0's
        zeros_count = len(s) - ones_count
        
        # Construct the maximum odd binary number:
        # 1. Place (ones_count-1) '1's at the leftmost positions
        # 2. Place all zeros in the middle
        # 3. Place the remaining '1' at the rightmost position to ensure odd
        return '1' * (ones_count - 1) + '0' * zeros_count + '1'