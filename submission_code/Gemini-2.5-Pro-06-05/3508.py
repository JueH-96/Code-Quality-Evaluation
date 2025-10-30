class Solution:
    def minChanges(self, n: int, k: int) -> int:
        """
        Calculates the minimum number of bit changes (1 to 0) to make n equal to k.
        """
        
        # Step 1: Check if the transformation is possible.
        # We can only change a '1' bit in `n` to a '0'. We can't change a '0' to a '1'.
        # This means that for any bit position, if `k` has a '1', `n` must also have a '1'.
        # In other words, the set bits of `k` must be a subset of the set bits of `n`.
        # This can be verified with the bitwise AND operation: `(n & k)` must be equal to `k`.
        # If not, it's impossible to form k from n.
        if (n & k) != k:
            return -1
        
        # Step 2: Calculate the number of changes.
        # If possible, the number of changes is the count of bits that are '1' in `n`
        # but '0' in `k`. These are precisely the bits we need to flip.
        #
        # The bitwise XOR operation (`^`) gives a number with '1's at all bit positions
        # where `n` and `k` differ. Since we've already confirmed the possibility,
        # all differing bits must be positions where `n` is '1' and `k` is '0'.
        # So, counting the set bits in `n ^ k` gives the number of required changes.
        
        # `bin(integer).count('1')` is a standard way to get the popcount in Python.
        # For Python 3.10+, `(n ^ k).bit_count()` would be more efficient.
        return bin(n ^ k).count('1')