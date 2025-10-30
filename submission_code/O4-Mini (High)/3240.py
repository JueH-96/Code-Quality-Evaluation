class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        # Compute the cumulative price sum f(N) = sum_{i=1..N} price(i)
        # Price(i) = count of set bits at positions that are multiples of x.
        def cumulative_price(N: int) -> int:
            total = 0
            max_bit = N.bit_length()
            # iterate over bit positions t = x, 2x, 3x, ... <= max_bit
            t = x
            while t <= max_bit:
                period = 1 << t
                half = period >> 1  # number of ones per full period at bit t
                full_periods = N // period
                rem = N % period
                # ones from full periods + ones from the partial (remainder) period
                total += full_periods * half + max(0, rem - half + 1)
                t += x
            return total

        # Binary search for the largest N such that cumulative_price(N) <= k
        lo, hi = 0, k * x
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if cumulative_price(mid) <= k:
                lo = mid
            else:
                hi = mid - 1
        return lo