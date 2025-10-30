class Solution:
    def isFascinating(self, n: int) -> bool:
        s = str(n) + str(2 * n) + str(3 * n)
        if len(s) != 9:
            return False
        digits = set(s)
        if '0' in digits:
            return False
        if len(digits) != 9:
            return False
        for i in range(1, 10):
            if str(i) not in s:
                return False
        return True