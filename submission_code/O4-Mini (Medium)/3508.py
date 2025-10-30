class Solution:
    def minChanges(self, n: int, k: int) -> int:
        # If k has any bit set that n does not, it's impossible
        if (k & ~n) != 0:
            return -1
        # We only need to turn off the bits that n has extra (bits in n but not in k)
        extra = n & ~k
        # Count how many 1s are in extra
        return extra.bit_count()