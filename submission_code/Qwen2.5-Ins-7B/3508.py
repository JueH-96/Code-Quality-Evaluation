class Solution:
    def minChanges(self, n: int, k: int) -> int:
        n_bin, k_bin = bin(n)[2:], bin(k)[2:]
        max_len = max(len(n_bin), len(k_bin))
        n_bin = n_bin.zfill(max_len)
        k_bin = k_bin.zfill(max_len)
        
        changes = 0
        for i in range(max_len):
            if n_bin[i] != k_bin[i]:
                changes += 1
        
        return changes if n_bin.replace('1', '0') >= k_bin else -1