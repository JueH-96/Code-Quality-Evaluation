class Solution:
    def minChanges(self, n: int, k: int) -> int:
        # If k has any 1 bit where n has 0 bit, it's impossible
        if (n & k) != k:
            return -1
        
        # Count the number of 1 bits in n that need to be changed to 0
        # These are the positions where n has 1 and k has 0
        return bin(n ^ k).count('1')