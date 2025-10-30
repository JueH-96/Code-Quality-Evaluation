class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        # Helper function to compute the total "price" for all numbers 1 through n.
        # For each number, its price is defined as the count of set bits in its 
        # binary representation (with bits numbered from right to left, starting at 1)
        # that occur in positions i where i % x == 0.
        #
        # Rather than iterating over each number, we compute the contribution
        # of a fixed bit-position i (1-indexed) among all numbers 1..n by using 
        # the formula for counting ones in a fixed position over a range:
        #
        #   countOnes(n, i) = (n // (2^i)) * (2^(i-1)) + max(0, (n % (2^i)) - 2^(i-1) + 1)
        #
        # We only need to sum this for bit positions i that are multiples of x.
        def total_price(n: int) -> int:
            if n <= 0:
                return 0
            total = 0
            # The number of bits needed to represent n in binary.
            bit_length = n.bit_length()
            # Iterate over all positions i that are multiples of x (i.e. i = x, 2x, 3x, ...)
            # but only up to the bit-length of n (since higher positions are all 0 for n).
            pos = x
            while pos <= bit_length:
                period = 1 << pos        # This is 2^pos.
                half_period = 1 << (pos - 1)  # This is 2^(pos-1); the run-length of bits set in each period.
                # The contribution for bit position "pos":
                total += (n // period) * half_period
                rem = n % period
                if rem >= half_period:
                    total += rem - half_period + 1
                pos += x
            return total

        # We need to determine the greatest integer num (n) such that:
        #   total_price(n) <= k
        # Since total_price(n) is a non-decreasing function, we can employ binary search.
        lo, hi = 1, 1
        # First, we find an upper bound "hi" such that total_price(hi) > k.
        while total_price(hi) <= k:
            hi *= 2

        ans = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if total_price(mid) <= k:
                ans = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return ans