class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        """
        Binary-search the largest integer n such that
        Σ_{i = 1..n} price(i) ≤ k,
        where  price(i)  is the number of set bits that are situated at
        positions which are multiples of x (1-indexed from the least
        significant bit).

        For a fixed bit position p (1-indexed) the bit is 1 exactly
        half of the time in every block of length 2ᵖ.  Therefore
            cnt(n, p) = number of integers in [1, n] whose p-th bit is 1
                      = (n+1)//2ᵖ · 2^{p-1} + max(0, (n+1) mod 2ᵖ − 2^{p-1})
        The cumulative price up to n is the sum of cnt(n, p) for all
        p that are multiples of x.
        """
        # -------------------------------------------------------------
        # helper: cumulative price up to n
        # -------------------------------------------------------------
        def cumulative(n: int) -> int:
            res = 0
            p = x                              # first relevant bit index
            while True:
                half = 1 << (p - 1)            # 2^{p-1}
                if half > n:                   # no number ≤ n contains this bit
                    break
                cycle = half << 1              # 2^{p}
                full_cycles = (n + 1) // cycle
                res += full_cycles * half
                remainder = (n + 1) % cycle
                if remainder > half:
                    res += remainder - half
                p += x
            return res
        # -------------------------------------------------------------
        # binary search on the answer
        # -------------------------------------------------------------
        lo, hi = 0, 1
        while cumulative(hi) <= k:             # grow upper bound
            hi <<= 1

        while lo < hi:                         # standard upper-bound binary search
            mid = (lo + hi) // 2
            if cumulative(mid) <= k:
                lo = mid + 1
            else:
                hi = mid
        return lo - 1