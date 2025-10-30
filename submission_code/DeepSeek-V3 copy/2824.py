class Solution:
    def isFascinating(self, n: int) -> bool:
        concatenated = str(n) + str(2 * n) + str(3 * n)
        if '0' in concatenated:
            return False
        digits = set(concatenated)
        return len(digits) == 9 and len(concatenated) == 9