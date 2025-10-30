class Solution:
    def minChanges(self, n: int, k: int) -> int:
        # Check if k is a subset of n in terms of set bits
        if (n & k) != k:
            return -1
        
        # Count the number of different bits (where n has 1 and k has 0)
        return bin(n ^ k).count('1')