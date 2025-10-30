class Solution:
    def minChanges(self, n: int, k: int) -> int:
        changes = 0

        temp_n = n
        temp_k = k

        while temp_n > 0 or temp_k > 0:
            n_bit = temp_n & 1
            k_bit = temp_k & 1

            if k_bit == 1 and n_bit == 0:
                return -1
            elif k_bit == 0 and n_bit == 1:
                changes += 1

            temp_n >>= 1
            temp_k >>= 1

        return changes