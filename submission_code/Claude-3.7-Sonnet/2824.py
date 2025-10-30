class Solution:
    def isFascinating(self, n: int) -> bool:
        # Calculate 2*n and 3*n
        doubled = 2 * n
        tripled = 3 * n
        
        # Concatenate the numbers as a string
        concatenated = str(n) + str(doubled) + str(tripled)
        
        # Check if the resulting string has exactly 9 characters
        # and contains all digits 1-9 exactly once
        return set(concatenated) == set('123456789') and len(concatenated) == 9