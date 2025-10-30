class Solution:
    def minChanges(self, n: int, k: int) -> int:
        if n < k:
            return -1
        
        bin_n = bin(n)[2:]
        bin_k = bin(k)[2:]
        
        # Pad the binary representations to have the same length
        max_len = max(len(bin_n), len(bin_k))
        bin_n = bin_n.zfill(max_len)
        bin_k = bin_k.zfill(max_len)
        
        changes = 0
        for i in range(max_len):
            if bin_n[i] == '1' and bin_k[i] == '0':
                changes += 1
            elif bin_n[i] == '0' and bin_k[i] == '1':
                return -1
        
        return changes