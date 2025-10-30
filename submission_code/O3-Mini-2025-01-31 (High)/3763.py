from bisect import bisect_right
from typing import List

class Solution:
    def separateSquares(self, squares: List[List[int]]) -> float:
        # For each square, we only need its vertical position "y" (the bottom)
        # and its side length l. The square spans vertically from y to y+l.
        # The area "below" a horizontal line at height L contributed by a square is:
        #   • 0 if L <= y,
        #   • l*(L - y) if y < L < y+l,
        #   • l*l if L >= y+l.
        #
        # To compute the total "below" area fast for any given L, we derive:
        #   f(L) = Σ_{s: y ≤ L} l * (L - y)    (but with the cap that if L > y+l, the
        #         contribution is only l*l instead of l*(L-y)).
        #
        # We can “adjust” the sum by noticing that for a square with top t = y+l,
        # if L >= t, the naive term would be: l*(L - y) = l*( (L - t) + l ) = l*(L-t) + l*l.
        # So if we first sum over all squares with bottom ≤ L
        #    B(L) = L * (Σ_{s with y ≤ L} l) - Σ_{s with y ≤ L} (l * y)
        # then in B(L) the “fully below” squares (those with t ≤ L) have been over‐counted.
        # For each square with t ≤ L, the excess is: l*(L-t).
        # Thus if we define
        #    F(L) = Σ_{s with t ≤ L} l * (L - (y+l)) = L * (Σ_{s with t ≤ L} l) - Σ_{s with t ≤ L} (l*(y+l)),
        # then the “correct” total below‐area is:
        #    f(L) = B(L) - F(L).
        #
        # Since all squares’ areas are counted fully (l*l) or partially (l*(L-y)), we
        # also have the total area S = Σ (l*l) and we need the horizontal line L such that:
        #    f(L) = S/2.
        #
        # f(L) is continuous and monotonic increasing. We can therefore locate the unique
        # L (within acceptable tolerance) by binary search over L.
        #
        # To perform the binary search quickly we precompute two sorted lists:
        #   1. Sorted by bottom y (for use in B(L)).
        #   2. Sorted by top t = y+l (for use in F(L)).
        # And we precompute the corresponding prefix sums.
        
        total_area = 0
        bottoms = []  # List of tuples: (y, l)
        tops = []     # List of tuples: (y+l, l)
        for _, y, l in squares:
            total_area += l * l
            bottoms.append((y, l))
            tops.append((y + l, l))
        
        bottoms.sort(key=lambda x: x[0])  # sort by bottom y
        tops.sort(key=lambda x: x[0])       # sort by top (y+l)
        
        # Build arrays and prefix sums for the squares sorted by bottom.
        bottoms_ys = []      # Sorted list of bottom y values.
        prefix_bottom_l = [0.0]   # prefix sum of l for squares with bottom ≤ some index.
        prefix_bottom_ly = [0.0]  # prefix sum of (l*y) for squares with bottom ≤ some index.
        for b, l in bottoms:
            bottoms_ys.append(b)
            prefix_bottom_l.append(prefix_bottom_l[-1] + l)
            prefix_bottom_ly.append(prefix_bottom_ly[-1] + b * l)
        
        # Build arrays and prefix sums for the squares sorted by top (y+l).
        tops_t = []          # Sorted list of top values (y+l).
        prefix_top_l = [0.0]    # prefix sum of l for squares with top ≤ some index.
        prefix_top_lt = [0.0]   # prefix sum of (l*(y+l)) for squares with top ≤ some index.
        for t, l in tops:
            tops_t.append(t)
            prefix_top_l.append(prefix_top_l[-1] + l)
            prefix_top_lt.append(prefix_top_lt[-1] + t * l)
        
        target = total_area / 2.0
        
        # The search space for the horizontal line L is in [min(bottom), max(top)].
        low = float(bottoms_ys[0])
        high = float(tops_t[-1])
        tol = 1e-7
        
        # Binary search for the smallest L such that f(L) >= target.
        while high - low > tol:
            mid = (low + high) / 2.0
            
            # Compute B(mid): contribution from all squares whose bottom is ≤ mid.
            idx_b = bisect_right(bottoms_ys, mid)
            B_val = mid * prefix_bottom_l[idx_b] - prefix_bottom_ly[idx_b]
            
            # Compute F(mid): subtract the excess from squares that are fully below mid.
            idx_t = bisect_right(tops_t, mid)
            F_val = mid * prefix_top_l[idx_t] - prefix_top_lt[idx_t]
            
            f_val = B_val - F_val
            
            if f_val < target:
                low = mid
            else:
                high = mid
        
        return high