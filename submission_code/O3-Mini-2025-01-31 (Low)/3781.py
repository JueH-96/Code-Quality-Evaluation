from typing import List

class Solution:
    def maxDistance(self, side: int, points: List[List[int]], k: int) -> int:
        # We need to choose k points from "points" such that the minimum Manhattan
        # distance between any 2 of the chosen points is as large as possible.
        # We can binary search on the candidate minimum distance D. For each candidate D,
        # we need to check if there is a selection of k points with all pairwise Manhattan
        # distances at least D.
        # 
        # Because k is small (<=25) while points can be up to about 15*10^3, a greedy
        # selection algorithm is a common heuristic. In many selection problems with a
        # feasibility check on a metric space, a greedy selection (especially if we sort the
        # points in some natural order) can decide feasibility.
        #
        # One common approach is to sort the points by a surrogate ordering (e.g. x+y)
        # and then iterate, choosing a point if it is at the required manhattan distance from
        # all previously chosen points. If we can get k points, the candidate D is feasible.
        #
        # The Manhattan distance between points [x1, y1] and [x2, y2] is:
        #    |x1-x2| + |y1-y2|
        #
        # The maximum distance between any two points on the boundary of a square with side length side
        # is 2*side.
        #
        # We then binary search on D from low=0 to high = 2*side.
        
        def can_select(D: int) -> bool:
            count = 0
            selected = []
            # sort points in ascending order based on x+y (a heuristic ordering)
            for p in sorted_points:
                # check if p is at least D away (manhattan) from every chosen point
                valid = True
                for sx, sy in selected:
                    if abs(p[0]-sx) + abs(p[1]-sy) < D:
                        valid = False
                        break
                if valid:
                    selected.append(p)
                    count += 1
                    if count >= k:
                        return True
            return False
        
        # sort points initially using a heuristic ordering.
        sorted_points = sorted(points, key=lambda p: (p[0] + p[1], p[0], p[1]))
        
        lo, hi = 0, 2 * side  # candidate answer in [0, 2*side]
        ans = 0
        
        while lo <= hi:
            mid = (lo + hi) // 2
            if can_select(mid):
                ans = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return ans