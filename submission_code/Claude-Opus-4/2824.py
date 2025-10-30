class Solution:
    def isFascinating(self, n: int) -> bool:
        # Concatenate n, 2*n, and 3*n
        concatenated = str(n) + str(2 * n) + str(3 * n)
        
        # Check if it contains exactly the digits 1-9
        # First check if length is 9 (for efficiency)
        if len(concatenated) != 9:
            return False
        
        # Check if it contains no 0s
        if '0' in concatenated:
            return False
        
        # Check if all digits 1-9 appear exactly once
        digit_set = set(concatenated)
        return len(digit_set) == 9 and digit_set == set('123456789')