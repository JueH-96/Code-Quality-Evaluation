class Solution:
    def minChanges(self, n: int, k: int) -> int:
        if n < k:
            return -1
        
        n_bin = bin(n)[2:]
        k_bin = bin(k)[2:]
        
        len_diff = len(n_bin) - len(k_bin)
        k_bin = '0' * len_diff + k_bin
        
        changes = 0
        for i in range(len(n_bin)):
            if n_bin[i] != k_bin[i] and n_bin[i] == '1':
                changes += 1
            elif n_bin[i] != k_bin[i] and n_bin[i] == '0':
                return -1
        
        return changes