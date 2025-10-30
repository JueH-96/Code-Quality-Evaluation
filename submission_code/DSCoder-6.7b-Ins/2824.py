class Solution:
    def isFascinating(self, n: int) -> bool:
        s = str(n) + str(2 * n) + str(3 * n)
        return set(s) == set(str(i) for i in range(1, 10)) and '0' not in s