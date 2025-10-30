class Solution:
    def minChanges(self, n: int, k: int) -> int:
        changes = 0
        for i in range(32):
            n_bit = (n >> i) & 1
            k_bit = (k >> i) & 1
            if n_bit == 1 and k_bit == 0:
                changes += 1
            elif n_bit == 0 and k_bit == 1:
                return -1
        return changes