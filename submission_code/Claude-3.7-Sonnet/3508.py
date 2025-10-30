class Solution:
    def minChanges(self, n: int, k: int) -> int:
        # If k has a 1 where n has a 0, it's impossible
        if (n ^ k) & k:
            return -1
        
        # Count the number of positions where n has a 1 and k has a 0
        val = n & ~k
        count = 0
        while val:
            count += val & 1
            val >>= 1
        
        return count