class Solution:
    def minChanges(self, n: int, k: int) -> int:
        mask = (1 << 20) - 1
        if k & (~n & mask):
            return -1
        flip = n & (~k & mask)
        return bin(flip).count('1')