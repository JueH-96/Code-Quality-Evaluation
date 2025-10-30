from bisect import bisect_right
from typing import List

class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        # Sort the given segments by their starting coordinate.
        coins.sort(key=lambda x: x[0])
        n = len(coins)
        # "segs" will be the sorted segments.
        segs = coins
        
        # Build two arrays:
        # prefix[i] = total coins in fully–covered segments among the first i segments.
        # ends[i] = ending coordinate of segs[i]
        prefix = [0] * (n + 1)
        ends = [0] * n
        for i, (l, r, c) in enumerate(segs):
            prefix[i + 1] = prefix[i] + c * (r - l + 1)
            ends[i] = r

        # Helper: F(x) returns total coins collected in all bags from coordinate 1 up to x.
        # Note: All coins are only in bags with coordinates >= 1, so for x < 1 we return 0.
        def F(x: int) -> int:
            if x < 1:
                return 0
            # Find how many segments have ended by x.
            idx = bisect_right(ends, x)
            total = prefix[idx]
            # If x falls into a segment that is not completely finished,
            # add the coins collected from that segment partially.
            if idx < n:
                l, r, c = segs[idx]
                if l <= x <= r:
                    total += c * (x - l + 1)
            return total

        # The answer will be the maximum coins obtainable from some k consecutive bags.
        # Idea: The summed coin value for a window [x, x+k-1] equals F(x+k-1) - F(x-1).
        # Because the coin distribution is piecewise constant (only changes at segment boundaries),
        # it suffices to consider candidate starting positions x that come from the boundaries.
        #
        # For each segment [l, r, c], the two kinds of breakpoints that can affect the sum are:
        #   1. When the left end of the window is “aligned” with a segment boundary.
        #   2. When the right end (x+k-1) is exactly a segment start or end.
        #
        # So we add:
        # - x = l            (window starts at a segment’s beginning)
        # - x = r + 1        (window starts just after a segment’s end)
        # - x = l - k + 1    (so that x+k-1 == l, the start of a segment)
        # - x = r - k + 1    (so that x+k-1 == r, the end of a segment)
        # Note: Some candidates might be out of any interesting range, so we will filter them.
        cand = set()
        for l, r, c in segs:
            cand.add(l)
            cand.add(r + 1)
            cand.add(l - k + 1)
            cand.add(r - k + 1)
        
        # Determine the range of x that may yield a nonzero sum.
        # If the window is completely to the left of all coins (bags < segs[0][0])
        # or completely to the right (starting x > segs[-1][1]),
        # it will have 0 coins.
        min_l = segs[0][0]
        max_r = segs[-1][1]
        lower_bound = min_l - k + 1  # if x < lower_bound then window's right end is before min_l.
        upper_bound = max_r          # if x > max_r then window starts after all coin–filled bags.
        
        ans = 0
        # Evaluate the k-length window for each candidate starting position.
        for x in cand:
            if x < lower_bound or x > upper_bound:
                continue
            window_end = x + k - 1
            # The coins in window [x, x+k-1] is F(x+k-1) - F(x-1)
            total_coins = F(window_end) - F(x - 1)
            if total_coins > ans:
                ans = total_coins
        return ans