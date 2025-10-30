class Solution:
    def isFascinating(self, n: int) -> bool:
        concatenated = str(n) + str(2 * n) + str(3 * n)
        if len(concatenated) != 9:
            return False
        digits = set(concatenated)
        if '0' in digits:
            return False
        if len(digits) != 9:
            return False
        return True