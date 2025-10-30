class Solution:
    def minChanges(self, n: int, k: int) -> int:
        # Check if k has any 1 bits where n has 0 bits
        # This would make it impossible to transform n to k
        if k & ~n:
            return -1
        
        # Count the number of 1 bits in n that need to be changed to 0
        # These are positions where n has 1 and k has 0
        return bin(n & ~k).count('1')