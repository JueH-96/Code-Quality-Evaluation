class Solution:
    def minEnd(self, n: int, x: int) -> int:
        """
        Let k = n - 1.  
        We build the array starting from x itself (the smallest number that surely
        keeps all the 1-bits required by the final AND).

        Every element of the array can be written as
            value = x | w
        where w uses ONLY the bit positions in which x has a 0
        (otherwise the AND would drop a 1-bit of x).

        To obtain the smallest possible last element we have to take the first `n`
        such numbers, i.e. the `n` smallest possible w’s.  
        Therefore `w_last` is the (n-1)-th non–negative integer whose 1-bits never
        overlap 1-bits of x.  
        This is exactly the number obtained by writing `k = n-1` in binary and
        inserting its bits, from least significant to most significant, into the
        zero positions of x (also from least significant upwards).

        The following loop performs that bit insertion directly.
        Complexity: O(number of bits) ≤ O(60).
        """
        k = n - 1           # how many extra numbers after x we need
        res = x             # will become the answer (last element)
        bit = 1             # current bit position we are looking at
        
        # Process until every bit of k has been placed
        while k:
            # skip positions where x already has a 1 (they cannot be used)
            if x & bit:
                bit <<= 1
                continue
            
            # place the current least-significant bit of k into this free position
            if k & 1:
                res |= bit
            k >>= 1          # move to the next bit of k
            bit <<= 1        # move to the next bit position in the number
        
        return res