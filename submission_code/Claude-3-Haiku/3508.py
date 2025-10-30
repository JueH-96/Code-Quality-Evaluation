class Solution:
    def minChanges(self, n: int, k: int) -> int:
        if n == k:
            return 0
        
        # Convert n and k to binary strings
        n_bin = bin(n)[2:]
        k_bin = bin(k)[2:]
        
        # Pad the shorter binary string with leading zeros
        if len(n_bin) < len(k_bin):
            n_bin = "0" * (len(k_bin) - len(n_bin)) + n_bin
        elif len(k_bin) < len(n_bin):
            k_bin = "0" * (len(n_bin) - len(k_bin)) + k_bin
        
        # Count the number of changes needed
        changes = 0
        for i in range(len(n_bin)):
            if n_bin[i] == "1" and n_bin[i] != k_bin[i]:
                changes += 1
        
        # If it's impossible to make n equal to k, return -1
        if changes > len(n_bin):
            return -1
        else:
            return changes