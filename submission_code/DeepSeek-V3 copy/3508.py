class Solution:
    def minChanges(self, n: int, k: int) -> int:
        if n == k:
            return 0
        if n < k:
            return -1
        # Check if all set bits in k are also set in n
        if (n & k) != k:
            return -1
        # Count the number of set bits in n that are not in k
        diff = n ^ k
        count = 0
        while diff:
            count += diff & 1
            diff >>= 1
        return count