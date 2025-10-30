class Solution:
    def isFascinating(self, n: int) -> bool:
        concatenated = str(n) + str(2 * n) + str(3 * n)
        return set(concatenated) == set('123456789') and len(concatenated) == 9