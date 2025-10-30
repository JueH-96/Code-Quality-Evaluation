class Solution:
    def minChanges(self, n: int, k: int) -> int:
        # If k is greater than n, it's impossible to make n equal to k
        if k > n:
            return -1
        
        # Count the number of bits that are 1 in n but not in k
        changes_needed = 0
        for i in range(21):  # Up to 2^20 > 10^6
            n_bit = (n >> i) & 1
            k_bit = (k >> i) & 1
            
            if n_bit == 1 and k_bit == 0:
                changes_needed += 1
            elif n_bit == 0 and k_bit == 1:
                # If k has a bit set that n does not, it's impossible
                return -1
        
        return changes_needed