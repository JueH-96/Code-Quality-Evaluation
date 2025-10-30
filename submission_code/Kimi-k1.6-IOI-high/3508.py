class Solution:
    def minChanges(self, n: int, k: int) -> int:
        if (n & k) != k:
            return -1
        xor = n ^ k
        return bin(xor).count('1')