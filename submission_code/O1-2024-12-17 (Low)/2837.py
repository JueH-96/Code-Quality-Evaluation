class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        # We want to find the minimum k (number of operations) such that:
        #
        #   num1 - k * num2 = sum_of_2_powers
        #
        # where sum_of_2_powers is a sum of exactly k terms, each of which is 2^i (0 <= i <= 60).
        #
        # Let x = num1 - k * num2. We need:
        #
        #   1) x >= 0  (since 2^i are nonnegative and we need the sum to match x)
        #   2) The popcount of x (the number of set bits in x) <= k
        #      (because to represent x as a sum of powers of 2, we need at least popcount(x) terms)
        #   3) k <= x  (because using k terms of powers of 2, the smallest sum is k*1 = k; it must not exceed x)
        #
        # We search for the smallest k satisfying these conditions. We'll limit k up to 60 or 61
        # because each operation can only subtract one power of two at a time, and the popcount
        # of any 64-bit integer cannot exceed 64. Empirically, going up to k=60 (or 61) is sufficient.

        import math

        # A small helper to count the set bits (popcount) in x
        def popcount(x: int) -> int:
            return x.bit_count()  # Python 3.10+; alternatively, bin(x).count('1')

        # Try k from 1 to 60 (inclusive). Feel free to go to 61 if desired.
        for k in range(1, 61):
            x = num1 - k * num2
            if x < 0:
                # If num2 > 0, x will only get smaller for larger k, so break early.
                if num2 > 0:
                    break
                # If num2 <= 0, x increases or stays the same sign as k grows, so just continue.
                continue

            # Check condition k <= x
            if k > x:
                continue

            # Check if popcount(x) <= k
            if popcount(x) <= k:
                return k

        return -1