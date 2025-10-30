class Solution:
    def minChanges(self, n: int, k: int) -> int:
        count = 0
        for i in range(32):
            n_bit = (n >> i) & 1
            k_bit = (k >> i) & 1
            if k_bit == 1 and n_bit == 0:
                return -1
            elif n_bit == 1 and k_bit == 0:
                count += 1
        return count