class Solution:
    def minChanges(self, n: int, k: int) -> int:
        if n == k:
            return 0
        if n < k:
            return -1

        binary_n = bin(n)[2:]
        binary_k = bin(k)[2:]
        
        len_n = len(binary_n)
        len_k = len(binary_k)
        
        max_len = max(len_n, len_k)
        
        binary_n = binary_n.zfill(max_len)
        binary_k = binary_k.zfill(max_len)

        changes = 0
        possible = True
        for i in range(max_len):
            if binary_k[i] == '1' and binary_n[i] == '0':
                possible = False
                break
            elif binary_k[i] == '0' and binary_n[i] == '1':
                changes += 1

        if possible:
            return changes
        else:
            return -1