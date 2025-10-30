class Solution:
    def minChanges(self, n: int, k: int) -> int:
        if (k & n) != k:
            return -1
        xor_result = n ^ k
        return bin(xor_result).count('1')