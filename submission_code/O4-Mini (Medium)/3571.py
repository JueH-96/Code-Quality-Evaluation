from typing import List
import bisect

class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        # The target point
        tx, ty = coordinates[k]
        
        # Split points into those strictly "left-below" and "right-above" of (tx, ty)
        left_pts = []
        right_pts = []
        for x, y in coordinates:
            if x < tx and y < ty:
                left_pts.append((x, y))
            elif x > tx and y > ty:
                right_pts.append((x, y))
        
        # A helper to compute the length of the longest strictly increasing chain
        # in both x and y, among a list of points.
        # We sort by x ascending, and for equal x by y descending (to forbid chaining
        # points with equal x), then do a standard LIS on the y-coordinates.
        def lis_2d(points: List[tuple]) -> int:
            # Sort by x ascending, y descending
            points.sort(key=lambda p: (p[0], -p[1]))
            tails = []
            for _, y in points:
                # find place to insert y into tails for a strictly increasing subsequence
                idx = bisect.bisect_left(tails, y)
                if idx == len(tails):
                    tails.append(y)
                else:
                    tails[idx] = y
            return len(tails)
        
        # LIS among points before the target
        L1 = lis_2d(left_pts)
        # LIS among points after the target
        L2 = lis_2d(right_pts)
        
        # Total path length is left-chain + the target itself + right-chain
        return L1 + 1 + L2