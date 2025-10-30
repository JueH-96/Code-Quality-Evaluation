class Solution:
    def minEnd(self, n: int, x: int) -> int:
        # If there's only one number, it must be x itself.
        if n == 1:
            return x
        
        # We need to pick n distinct numbers a_i = x | y_i, strictly increasing,
        # where y_i & x == 0 and y_0 = 0.  The AND of all a_i will be x.
        # The smallest possible maximum is obtained by picking the first n
        # nonnegative integers y that avoid bits of x, i.e. y & x == 0, in increasing order.
        #
        # Those y form an isomorphic sequence to the nonnegative integers k = 0,1,2,...
        # by mapping the j-th zero-bit position of x to the j-th bit of k.
        # Hence the (n-1)-th y is obtained by writing k = n-1 in binary
        # and placing its j-th bit into the j-th zero position of x.
        
        k = n - 1
        # Determine how many bits of k we have
        L = k.bit_length()  # if k=0, bit_length=0, but we still need one zero position to map 0->0
        if L == 0:
            L = 1
        
        # Collect the positions of zero-bits in x, in ascending order,
        # until we have at least L of them.
        zero_positions = []
        pos = 0
        while len(zero_positions) < L:
            if (x & (1 << pos)) == 0:
                zero_positions.append(pos)
            pos += 1
        
        # Build y by mapping bits of k into those zero positions
        y = 0
        for j in range(L):
            if (k >> j) & 1:
                y |= (1 << zero_positions[j])
        
        return x | y


# Example usage:
# sol = Solution()
# print(sol.minEnd(3, 4))  # -> 6
# print(sol.minEnd(2, 7))  # -> 15