class Solution:
    def minChanges(self, n: int, k: int) -> int:
        # Check if k has any bit set that n does not have set
        if (k & ~n) != 0:
            return -1
        
        # Count how many bits are set in n but not in k
        return bin(n & ~k).count('1')