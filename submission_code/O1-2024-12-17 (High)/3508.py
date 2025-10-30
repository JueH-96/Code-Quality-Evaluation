class Solution:
    def minChanges(self, n: int, k: int) -> int:
        # Check if every bit of k is also set in n (k must be a submask of n).
        if (n & k) != k:
            return -1
        
        # Count how many bits are set in n but not in k.
        # These bits must be flipped from 1 to 0.
        return bin(n & ~k).count('1')