class Solution:
    def minChanges(self, n: int, k: int) -> int:
        # Step 1: Check if it's possible to transform n into k.
        # We can only change '1' bits to '0' bits in n.
        # This implies that for any bit position 'i', if the i-th bit of k is '1',
        # the i-th bit of n must also be '1'. If n has a '0' where k has a '1',
        # it's impossible, as we cannot change a '0' to a '1'.
        # The condition (n & k) == k effectively checks this:
        # If (n & k) == k, it means all bits that are '1' in k are also '1' in n.
        # If (n & k) != k, it means there's at least one bit where k has a '1'
        # but n has a '0', making the transformation impossible.
        if (n & k) != k:
            return -1
        
        # Step 2: If the transformation is possible, count the number of changes needed.
        # Since we've already established that (n & k) == k, we know there are no
        # positions where n has a '0' and k has a '1'.
        # Therefore, any difference between n and k must come from positions where
        # n has a '1' and k has a '0'. These are precisely the bits that need to be
        # changed from '1' to '0'.
        
        # The bitwise XOR operation (n ^ k) produces a number where a bit is '1'
        # if the corresponding bits in n and k are different.
        # Given our prior check, all '1's in (n ^ k) will represent positions
        # where n had a '1' and k had a '0'.
        # Counting these '1's gives us the minimum number of necessary changes.
        return (n ^ k).bit_count()