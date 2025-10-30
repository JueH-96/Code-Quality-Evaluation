class Solution:
    def isFascinating(self, n: int) -> bool:
        concatenated = str(n) + str(2*n) + str(3*n)
        # Check if the concatenated string has the digits 1-9 exactly once
        return len(concatenated) == 9 and set(concatenated) == set("123456789")