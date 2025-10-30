class Solution:
    def isFascinating(self, n: int) -> bool:
        # Concatenate n, 2*n, and 3*n
        concatenated = str(n) + str(2 * n) + str(3 * n)
        
        # Check if it contains exactly digits 1-9
        return sorted(concatenated) == list('123456789')