class Solution:
    def minEnd(self, n: int, x: int) -> int:
        """
        We want an array nums of length n, strictly increasing (nums[i+1] > nums[i])
        and such that the bitwise AND of all elements equals x. We must return the
        minimum possible value of nums[n-1].
        
        Key idea:
        All elements must preserve the bits set in x (otherwise the AND would lose
        those bits).  Thus each element y satisfies y & x = x.  In other words,
        for every bit that is 1 in x, that bit is also 1 in y.

        Among all numbers y >= x satisfying y & x = x, we want to pick n distinct
        values in ascending order and look at the last one.  It turns out that the
        n-th smallest such number can be computed directly:

          - Let "d" be the n-th smallest nonnegative integer such that (d & x) == 0.
            (This ensures that setting those bits in 'd' does not conflict with the bits of 'x'.)
          - Then the n-th smallest valid number is simply x + d (equivalently x | d, since x & d = 0).

        To find that "d" for a given n in ascending order, observe that the integers
        whose bits do not overlap with x can be enumerated by taking the binary
        representation of (n-1) and mapping its bits into the zero-bit positions of x.
        
        Example: x=7 (binary 0b0111). The zero-bit positions start from bit 3 onward
        (3,4,5,...). If n=2 -> (n-1)=1 (binary 0b1). We set the first free bit (bit 3)
        to 1, so d=1<<3=8. Then x + d = 7 + 8 = 15, matching the example.
        
        This implementation runs in O(log n), which is sufficient for large n.
        """

        # If n=1, the answer is x itself, but our general logic covers that anyway
        if n == 1:
            return x

        # We want the (n-1)-th pattern of free bits (because for n distinct values,
        # we start counting from d=0 as the first, which corresponds to n=1).
        m = n - 1  
        
        d = 0
        bit_pos = 0  # We will scan bit positions from low to high
        while m > 0:
            # Skip positions where x has a '1' (because d & x must be 0 there)
            while (x & (1 << bit_pos)) != 0:
                bit_pos += 1
            # If the current LSB of m is 1, set this free bit in d
            if (m & 1) == 1:
                d |= (1 << bit_pos)
            m >>= 1
            bit_pos += 1
        
        # The final answer is x + d, since x & d = 0 => x + d = x | d
        return x + d