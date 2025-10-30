class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        """
        We want the minimum k such that after k operations, num1 becomes 0.
        Each operation subtracts (2^i + num2) for some i in [0..60].
        
        After k operations, the net subtraction from num1 is:
            k * num2  +  sum_of_powers_of_two
        and we want the result to be exactly 0. Denote:
            N = num1 - k * num2
        so we need:
            sum_of_powers_of_two = N

        Because each 2^i >= 1, if we use exactly k terms of the form 2^i,
        the sum is at least k (picking i=0 for each term) and at most k * 2^60
        (picking i=60 each time).  In addition, one can show that any N
        in the range [k, k*2^60] can be formed by k powers of 2 in [0..60]
        if and only if k is at least the number of 1-bits in N (popcount(N))
        and at most N.  (We can "split" larger powers of 2 if needed.)
        
        Hence, for each k, we check:
         1) N = num1 - k*num2 must be > 0.
         2) popcount(N) <= k <= N.

        We only need to try k up to 60 (or 61) because i goes only up to 60,
        and the worst-case popcount for a 61-bit number can be 61.  If no suitable
        k in that range works, the answer is -1.
        """
        
        import sys
        
        # We'll go up to 60 (or 61) inclusively
        # 61 is enough to cover the case of needing up to 61 bits (e.g. 2^61 - 1).
        for k in range(1, 62):
            N = num1 - k * num2
            if N <= 0:
                continue
            # Count the set bits (popcount). Python 3.10+ has int.bit_count()
            # Otherwise use: popcount = bin(N).count('1')
            popcount = N.bit_count()
            
            # Check if we can form N with k powers of 2 in [0..60].
            if popcount <= k <= N:
                return k
        
        return -1