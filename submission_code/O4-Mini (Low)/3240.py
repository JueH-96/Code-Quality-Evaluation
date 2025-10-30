class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        # Compute total price sum for all numbers from 1 to n
        def total_price(n: int) -> int:
            total = 0
            # consider bit positions p = x, 2x, 3x, ... up to ~60
            p = x
            while p <= 60:
                cycle = 1 << p           # 2^p
                half = 1 << (p - 1)      # 2^(p-1)
                full_cycles = n // cycle
                rem = n % cycle
                # in each full cycle, exactly half ones at position p
                cnt = full_cycles * half
                # plus the ones in the partial cycle
                # ones occur in the interval [half, cycle-1]
                if rem >= half:
                    cnt += rem - half + 1
                total += cnt
                p += x
            return total

        # binary search for maximum n such that total_price(n) <= k
        lo, hi = 0, 1 << 60
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if total_price(mid) <= k:
                lo = mid
            else:
                hi = mid - 1
        return lo