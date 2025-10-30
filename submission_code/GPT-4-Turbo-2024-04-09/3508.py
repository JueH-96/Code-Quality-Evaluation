class Solution:
    def minChanges(self, n: int, k: int) -> int:
        if n == k:
            return 0
        
        # Count the number of bits that need to be changed from 1 to 0 in n to match k
        changes_needed = 0
        while n > 0 or k > 0:
            bit_n = n & 1
            bit_k = k & 1
            
            if bit_n == 1 and bit_k == 0:
                changes_needed += 1
            elif bit_n == 0 and bit_k == 1:
                # If we need a bit to be 1 in k where it is 0 in n, it's impossible
                return -1
            
            n >>= 1
            k >>= 1
        
        return changes_needed