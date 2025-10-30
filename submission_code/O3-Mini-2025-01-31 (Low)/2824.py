class Solution:
    def isFascinating(self, n: int) -> bool:
        # Create concatenated string of n, 2*n, and 3*n
        concatenated = str(n) + str(2*n) + str(3*n)
        # It must be exactly 9 characters long.
        if len(concatenated) != 9:
            return False
        # Check that the concatenated string contains all digits from 1 to 9 exactly once
        digits = set(concatenated)
        return digits == set("123456789")