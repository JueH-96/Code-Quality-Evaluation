class Solution:
    def minChanges(self, n: int, k: int) -> int:
        n_bin = bin(n)[2:]
        k_bin = bin(k)[2:]
        n_len = len(n_bin)
        k_len = len(k_bin)

        if n_len > k_len:
            return -1

        n_bin = n_bin.zfill(k_len)
        k_bin = k_bin.zfill(k_len)

        changes = 0
        for i in range(k_len):
            if n_bin[i] != k_bin[i]:
                changes += 1

        return changes