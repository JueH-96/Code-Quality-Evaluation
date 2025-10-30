class Solution:
    def minChanges(self, n: int, k: int) -> int:
        if k > n:
            return -1
        n_bin = bin(n)[2:]
        k_bin = bin(k)[2:]
        n_bin = n_bin.zfill(max(len(n_bin), len(k_bin)))
        k_bin = k_bin.zfill(max(len(n_bin), len(k_bin)))
        changes = 0
        for i in range(len(n_bin)):
            if n_bin[i] == '1' and k_bin[i] == '0':
                changes += 1
        return changes