class Solution:
    def minChanges(self, n: int, k: int) -> int:
        # Check if all bits set in k are also set in n
        # This ensures it's possible to reach k by only turning bits off from n.
        if (n & k) != k:
            return -1
        
        # Count how many bits are set in n that are not set in k
        # These are the bits we need to flip from 1 to 0.
        diff_bits = n & (~k)
        
        # Return the count of set bits in diff_bits
        return bin(diff_bits).count('1')