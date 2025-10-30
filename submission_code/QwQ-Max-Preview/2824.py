class Solution:
    def isFascinating(self, n: int) -> bool:
        if n > 333:
            return False
        s = str(n) + str(2 * n) + str(3 * n)
        if '0' in s:
            return False
        return sorted(s) == list('123456789')