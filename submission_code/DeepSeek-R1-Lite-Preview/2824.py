class Solution:
    def isFascinating(self, n: int) -> bool:
        n2 = 2 * n
        n3 = 3 * n
        concatenated = str(n) + str(n2) + str(n3)
        return ''.join(sorted(concatenated)) == '123456789'