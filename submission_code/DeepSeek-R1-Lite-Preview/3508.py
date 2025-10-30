class Solution:
    def minChanges(self, n: int, k: int) -> int:
        changes = 0
        while n > 0 or k > 0:
            n_bit = n & 1
            k_bit = k & 1
            if n_bit == 1 and k_bit == 0:
                changes += 1
            elif n_bit == 0 and k_bit == 1:
                return -1
            n >>= 1
            k >>= 1
        return changes