class Solution:
    def minChanges(self, n: int, k: int) -> int:
        # Convert n and k to binary and remove the '0b' prefix
        bin_n = bin(n)[2:]
        bin_k = bin(k)[2:]
        
        # Make the binary strings the same length by padding with zeros
        max_len = max(len(bin_n), len(bin_k))
        bin_n = bin_n.zfill(max_len)
        bin_k = bin_k.zfill(max_len)
        
        # Initialize a counter for the number of changes needed
        changes = 0
        
        # Iterate over the binary strings from right to left
        for i in range(max_len - 1, -1, -1):
            # If the current bits are different and the bit in n is 1
            if bin_n[i] != bin_k[i] and bin_n[i] == '1':
                # Increment the changes counter
                changes += 1
            # If the current bits are different and the bit in n is 0
            elif bin_n[i] != bin_k[i] and bin_n[i] == '0':
                # It's impossible to make n equal to k
                return -1
        
        # Return the number of changes needed
        return changes