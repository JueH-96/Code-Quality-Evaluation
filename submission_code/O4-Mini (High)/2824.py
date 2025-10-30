class Solution:
    def isFascinating(self, n: int) -> bool:
        # Concatenate n, 2*n, and 3*n as strings
        s = str(n) + str(2 * n) + str(3 * n)
        
        # Must be exactly 9 digits and contain no '0'
        if len(s) != 9 or '0' in s:
            return False
        
        # Check if it contains all digits from '1' to '9' exactly once
        return set(s) == set('123456789')