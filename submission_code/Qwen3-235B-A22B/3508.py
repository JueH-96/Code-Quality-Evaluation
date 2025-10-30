class Solution:
    def minChanges(self, n: int, k: int) -> int:
        if k & (~n) != 0:
            return -1
        return bin(n ^ k).count('1')