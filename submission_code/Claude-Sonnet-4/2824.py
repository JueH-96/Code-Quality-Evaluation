class Solution:
    def isFascinating(self, n: int) -> bool:
        # Concatenate n, 2*n, and 3*n
        concatenated = str(n) + str(2 * n) + str(3 * n)
        
        # Check if the result has exactly 9 digits
        if len(concatenated) != 9:
            return False
        
        # Check if it contains no 0's and has all digits 1-9 exactly once
        digit_set = set(concatenated)
        
        # Should not contain '0' and should contain exactly digits '1' to '9'
        return '0' not in digit_set and digit_set == set('123456789')