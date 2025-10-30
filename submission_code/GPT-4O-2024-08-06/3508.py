class Solution:
    def minChanges(self, n: int, k: int) -> int:
        # If k is greater than n, it's impossible to make n equal to k by only changing 1s to 0s
        if k > n:
            return -1
        
        # Count the number of 1s in the binary representation of n and k
        n_ones = bin(n).count('1')
        k_ones = bin(k).count('1')
        
        # If k has more 1s than n, it's impossible to make n equal to k
        if k_ones > n_ones:
            return -1
        
        # The number of changes needed is the difference in the number of 1s
        return n_ones - k_ones