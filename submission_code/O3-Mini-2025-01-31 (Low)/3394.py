class Solution:
    def minEnd(self, n: int, x: int) -> int:
        # if n==1, the sequence only contains x.
        if n == 1:
            return x
        
        # We want to find the minimal d such that:
        #   d >= L, where L = n - 1, and
        #   for every bit i, if d has a 1 at bit i then x has 0 at bit i.
        # Then our answer is x + d.
        L = n - 1
        
        # Set number of bits M to consider.
        M = max(L.bit_length(), x.bit_length()) + 1
        
        # allowed[i] = True if we are allowed to set bit i in d (i.e. if x does not have that bit).
        # Bits are considered with 0 as the least significant.
        allowed = [False] * M
        for i in range(M):
            # if bit i in x is 0 then we can use 1 in d at that position.
            if ((x >> i) & 1) == 0:
                allowed[i] = True
        
        # We use memoization for our DP.
        from functools import lru_cache
        
        # f(pos, tight): for bits pos down to 0, return the minimal integer (for those bits, as an int)
        # that can be formed from allowed bits such that
        # if tight is True, then the prefix we have chosen equals the prefix of L for bit positions > pos.
        # In that case, the number we form must be >= the corresponding bits of L.
        # If no solution exists, return None.
        @lru_cache(maxsize=None)
        def f(pos, tight):
            if pos < 0:
                # All bits chosen
                return 0
            # Get the bit of L at position pos.
            Lbit = (L >> pos) & 1
            # We will try possible bits for d at this position.
            for b in (0, 1):
                # If b==1, then that position must be allowed.
                if b == 1 and not allowed[pos]:
                    continue
                # When tight is True, we must choose b such that b >= Lbit.
                if tight and b < Lbit:
                    continue
                new_tight = tight and (b == Lbit)
                suffix = f(pos - 1, new_tight)
                if suffix is not None:
                    # Build the number
                    return (b << pos) | suffix
            return None
        
        d = f(M - 1, True)
        # The required result:
        return x + d

# For testing purposes (you can delete or comment out the code below)
if __name__ == '__main__':
    sol = Solution()
    # Example 1: n = 3, x = 4; Expected output: 6
    print(sol.minEnd(3, 4))  # Output: 6
    # Example 2: n = 2, x = 7; Expected output: 15
    print(sol.minEnd(2, 7))  # Output: 15