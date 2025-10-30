class Solution:
    def isFascinating(self, n: int) -> bool:
        s = str(n) + str(2*n) + str(3*n)
        if '0' in s or len(s) != 9:
            return False
        return set(s) == set('123456789')