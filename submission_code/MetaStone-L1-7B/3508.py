class Solution:
    def minChanges(self, n: int, k: int) -> int:
        diff = n ^ k
        if (n & diff) != diff:
            return -1
        return bin(diff).count('1')