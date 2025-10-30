class Solution:
    def minChanges(self, n: int, k: int) -> int:
        # Convert both numbers to binary and align them to the rightmost bit
        bin_n = bin(n)[2:][::-1]
        bin_k = bin(k)[2:][::-1]
        
        max_length = max(len(bin_n), len(bin_k))
        bin_n = bin_n.ljust(max_length, '0')
        bin_k = bin_k.ljust(max_length, '0')
        
        # Count the number of bit changes needed to transform n into k
        changes = 0
        for i in range(max_length):
            if bin_n[i] == '1' and bin_k[i] != bin_n[i]:
                changes += 1
            elif bin_n[i] == '1' and bin_k[i] == '0' and i > 0 and bin_k[i-1] != '1':
                # If a '1' bit in n has to become '0' and is not preceded by a '1' in k, it's impossible
                return -1
                
        return changes