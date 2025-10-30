class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        # We want to find the smallest k >= 1 such that
        #   num1 - k * num2 = S
        # and S can be expressed as a sum of exactly k powers of two (repetitions allowed).
        #
        # Let S = num1 - k * num2.  We need:
        #   1) S >= k      (since the smallest sum of k powers of two is k*2^0 = k)
        #   2) popcount(S) <= k  (since if S in binary has b bits set, we need at least b summands)
        #   3) k <= S      (since the largest number of summands, all equal to 1, is S itself)
        #
        # If those hold, we can split the 1-bits of S further into 2^0's to reach exactly k terms.
        #
        # We only need to try k up to some reasonable bound. Empirically,
        # k never needs to exceed about 60 for num1 up to 1e9 and |num2| up to 1e9.
        # (Beyond that, either S < k or popcount(S) is too large initially.)
        
        def popcount(x: int) -> int:
            return x.bit_count()
        
        for k in range(1, 61):
            S = num1 - k * num2
            if S < k:
                # Too small to be represented as sum of k powers of two
                continue
            # Check bit-count constraint
            b = popcount(S)
            if b <= k <= S:
                return k
        
        return -1