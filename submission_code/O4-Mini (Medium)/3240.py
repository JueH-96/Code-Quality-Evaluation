class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        # Count of numbers in [1..n] whose p-th bit (1-indexed from LSB) is 1.
        def count_ones(n: int, p: int) -> int:
            # shift = p-1, cycle length = 2^(shift+1)
            shift = p - 1
            cycle = 1 << (shift + 1)
            # We consider numbers 0..n inclusive; zero contributes nothing.
            full_cycles = (n + 1) // cycle
            ones = full_cycles * (1 << shift)
            rem = (n + 1) % cycle
            # In the partial cycle, bits are 0 for the first half, then 1 for the next half
            ones += max(0, rem - (1 << shift))
            return ones

        # Total price of numbers 1..n
        def total_price(n: int) -> int:
            tot = 0
            p = x
            # Stop when that bit position is beyond n's max bit
            while (1 << (p - 1)) <= n:
                tot += count_ones(n, p)
                p += x
            return tot

        # Binary search for largest n such that total_price(n) <= k
        lo, hi = 0, 10**18
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if total_price(mid) <= k:
                lo = mid
            else:
                hi = mid - 1
        return lo