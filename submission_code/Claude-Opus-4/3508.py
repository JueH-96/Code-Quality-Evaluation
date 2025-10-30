class Solution:
    def minChanges(self, n: int, k: int) -> int:
        # If k has any bit set to 1 where n has 0, it's impossible
        # This can be checked using bitwise AND: if (k & ~n) != 0, then impossible
        if (k & ~n) != 0:
            return -1
        
        # Count the number of bits that need to be changed from 1 to 0
        # These are positions where n has 1 and k has 0
        # This is given by n XOR k, but only for positions where n has 1
        # Which is (n ^ k) & n, or simply n & ~k
        changes = n & ~k
        
        # Count the number of 1s in changes
        count = 0
        while changes:
            count += changes & 1
            changes >>= 1
        
        return count