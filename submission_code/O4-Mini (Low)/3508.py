class Solution:
    def minChanges(self, n: int, k: int) -> int:
        # If k has a 1-bit where n has a 0-bit, it's impossible
        if (n & k) != k:
            return -1
        # Count the 1-bits in n that need to be flipped to 0,
        # i.e., bits set in n but not in k
        to_flip = n & (~k)
        return to_flip.bit_count()