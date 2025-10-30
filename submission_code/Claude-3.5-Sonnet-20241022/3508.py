class Solution:
    def minChanges(self, n: int, k: int) -> int:
        # Convert n and k to binary strings
        n_bin = bin(n)[2:]
        k_bin = bin(k)[2:]
        
        # Make binary strings equal length by padding with zeros
        max_len = max(len(n_bin), len(k_bin))
        n_bin = n_bin.zfill(max_len)
        k_bin = k_bin.zfill(max_len)
        
        changes = 0
        
        # Check each bit position
        for i in range(max_len):
            if n_bin[i] == '1' and k_bin[i] == '0':
                changes += 1
            elif n_bin[i] == '0' and k_bin[i] == '1':
                # If n has 0 but k has 1, it's impossible
                return -1
        
        return changes