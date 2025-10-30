class Solution:
    def isFascinating(self, n: int) -> bool:
        # Concatenate n, 2*n, and 3*n
        concatenated = str(n) + str(2 * n) + str(3 * n)
        
        # Check if it contains all digits from 1 to 9 exactly once
        return set(concatenated) == set('123456789') and len(concatenated) == 9