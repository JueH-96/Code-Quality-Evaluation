class Solution:
    def minChanges(self, n: int, k: int) -> int:
        if n < k:
            return -1
        
        n_bin = bin(n)[2:]
        k_bin = bin(k)[2:]
        
        # Make sure both binary strings are of the same length
        max_len = max(len(n_bin), len(k_bin))
        n_bin = n_bin.zfill(max_len)
        k_bin = k_bin.zfill(max_len)
        
        changes_needed = 0
        for n_bit, k_bit in zip(reversed(n_bin), reversed(k_bin)):
            if n_bit == '1' and k_bit == '0':
                changes_needed += 1
            elif n_bit == '0' and k_bit == '1':
                return -1
        
        return changes_needed