class Solution:
    def minChanges(self, n: int, k: int) -> int:
        n_bin = bin(n)[2:]
        k_bin = bin(k)[2:]
        
        n_bits = [int(bit) for bit in n_bin]
        k_bits = [int(bit) for bit in k_bin]
        
        len_n = len(n_bits)
        len_k = len(k_bits)
        
        if len_k > len_n:
            return -1
        
        changes = 0
        
        for i in range(1, len_k + 1):
            if n_bits[len_n - i] == 1 and k_bits[len_k -i] == 0:
                changes += 1
        
        for i in range(len_n - len_k):
            if n_bits[i] == 1:
                changes +=1
                
        
        if n < k:
            return -1
            
        return changes