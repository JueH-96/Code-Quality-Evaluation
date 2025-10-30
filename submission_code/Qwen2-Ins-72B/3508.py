class Solution:
    def minChanges(self, n: int, k: int) -> int:
        # Convert both numbers to binary
        bin_n = bin(n)[2:]
        bin_k = bin(k)[2:]
        
        # Pad the shorter binary string with zeros on the left
        max_len = max(len(bin_n), len(bin_k))
        bin_n = bin_n.zfill(max_len)
        bin_k = bin_k.zfill(max_len)
        
        # Count the number of changes needed
        changes = 0
        for i in range(max_len):
            if bin_n[i] == '1' and bin_n[i] != bin_k[i]:
                changes += 1
                if bin_k[i] == '0':
                    # If the bit in k is 0 and the bit in n is 1, it's impossible to make n equal to k
                    return -1
        
        return changes