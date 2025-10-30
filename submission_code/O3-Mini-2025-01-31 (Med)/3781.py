from typing import List
import sys

class Solution:
    def maxDistance(self, side: int, points: List[List[int]], k: int) -> int:
        # Increase recursion limit (k is small but we might scan many points)
        sys.setrecursionlimit(10000)
        
        # Helper: compute Manhattan distance
        def manhattan(p1, p2):
            return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])
        
        # Compute "perimeter order" for the points.
        # The rule is: 
        #   if the point is on bottom edge (y==0): pos = x.
        #   if on right edge (x==side): pos = side + y.
        #   if on top edge (y==side): pos = side + (side) + (side - x) = 2*side + (side - x); 
        #       but we use convention pos = 3*side - x for simplicity.
        #   if on left edge (x==0): pos = 3*side + (side - y) = 4*side - y.
        def perim_pos(pt):
            x, y = pt
            if y == 0:
                return x
            elif x == side:
                return side + y
            elif y == side:
                return 3 * side - x  # note: 2*side + (side - x)
            elif x == 0:
                return 4 * side - y
            else:
                # should not happen because points are on boundary.
                return 0
        
        # sort points using the computed perimeter order
        sorted_points = sorted(points, key=perim_pos)
        n = len(sorted_points)

        # DFS feasibility check:
        # Check if it is possible to choose "k" points (from sorted_points starting at index "i")
        # such that every chosen point has Manhattan distance >= d from every other chosen point.
        def can_choose(i, chosen):
            # chosen: list of previously chosen point coordinates
            if len(chosen) == k:
                return True
            if n - i < k - len(chosen):
                return False
            # Try iterating over indices starting from i.
            for j in range(i, n):
                p = sorted_points[j]
                # check compatibility with every previously chosen point
                ok = True
                for q in chosen:
                    if manhattan(p, q) < d:
                        ok = False
                        break
                if not ok:
                    continue
                # choose p
                if can_choose(j + 1, chosen + [p]):
                    return True
            return False

        # When d <= 1 then since points are distinct integers on the boundary,
        # the Manhattan distance is always at least 1.
        def feasible(d):
            nonlocal sorted_points, n, k
            # For d <= 1, always true (if k <= n)
            if d <= 1:
                return True
            # Otherwise, use DFS/backtracking.
            return can_choose(0, [])
        
        # Binary search on [1, 2*side] for the maximum d that is feasible.
        low, high = 1, 2 * side + 1  # high is exclusive
        ans = 1
        while low < high:
            mid = (low + high) // 2
            d = mid
            if feasible(d):
                ans = d
                low = mid + 1
            else:
                high = mid
        return ans