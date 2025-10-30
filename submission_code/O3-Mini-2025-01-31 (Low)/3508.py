class Solution:
    def minChanges(self, n: int, k: int) -> int:
        # If n is already equal to k, no changes are needed.
        if n == k:
            return 0
        # We can only turn off bits, which means the resulting number will be <= n.
        # So if k > n, it is impossible.
        if k > n:
            return -1
        # Check the subset property:
        # All bits set in k must have been set in n.
        if (n & k) != k:
            return -1
        
        # Count the number of set bits in n and k.
        # The bits that need to be turned off are those which are 1 in n but 0 in k.
        # Thus the number of changes is the difference in their count of ones.
        count_n = bin(n).count("1")
        count_k = bin(k).count("1")
        
        return count_n - count_k