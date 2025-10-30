class Solution:
    def isFascinating(self, n: int) -> bool:
        # Form the concatenated string of n, 2*n, and 3*n
        s = str(n) + str(2 * n) + str(3 * n)
        # Check that it is exactly length 9 and contains all digits '1' to '9' exactly once
        return len(s) == 9 and set(s) == set('123456789')