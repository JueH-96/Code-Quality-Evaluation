class Solution:
    def isFascinating(self, n: int) -> bool:
        concatenated = str(n) + str(2 * n) + str(3 * n)
        digits = set(concatenated)
        return len(concatenated) == 9 and digits == set('123456789')