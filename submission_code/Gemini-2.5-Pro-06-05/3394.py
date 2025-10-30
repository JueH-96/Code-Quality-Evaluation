class Solution:
    def minEnd(self, n: int, x: int) -> int:
        """
        Constructs an array of positive integers `nums` of size `n` where for every
        0 <= i < n - 1, nums[i + 1] is greater than nums[i], and the result of
        the bitwise AND operation between all elements of nums is x.
        Returns the minimum possible value of nums[n - 1].
        """

        # The conditions imply that every number in the array must be "x-compatible",
        # meaning k & x == x. To minimize the last element, we must choose the first
        # n numbers from the sequence of x-compatible numbers.
        # This problem reduces to finding the n-th number in this sequence.

        # The n-th x-compatible number is constructed by taking x and setting additional
        # bits at positions where x has a 0. The bits to set are determined by the
        # binary representation of n - 1.
        
        # We use n-1 because the sequence is 0-indexed (the 1st number corresponds to 0).
        v = n - 1
        
        # The result starts as x. We will OR it with additional bits.
        res = x
        
        # bit_pos tracks the current bit position we are examining.
        bit_pos = 0
        
        # We continue as long as there are bits in v to be placed.
        while v > 0:
            # Check if the current bit position is "free" in x (i.e., the bit is 0).
            if (x >> bit_pos) & 1 == 0:
                # This position is free. We use it for the current LSB of v.
                if (v & 1) == 1:
                    # If the LSB of v is 1, set the corresponding free bit in the result.
                    res |= (1 << bit_pos)
                
                # We have consumed the LSB of v, so we move to its next bit.
                v >>= 1
            
            # Move to the next bit position.
            bit_pos += 1
            
        return res