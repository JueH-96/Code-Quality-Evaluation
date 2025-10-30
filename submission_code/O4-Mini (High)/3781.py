from typing import List

class Solution:
    def maxDistance(self, side: int, points: List[List[int]], k: int) -> int:
        # Map each boundary point to a parameter along the perimeter,
        # then sort so we can try greedy selections in this order.
        n = len(points)
        per4 = 4 * side
        pts = []
        for x, y in points:
            if y == 0:
                # bottom edge, from (0,0) to (side,0)
                p = x
            elif x == side:
                # right edge, from (side,0) to (side,side)
                p = side + y
            elif y == side:
                # top edge, from (side,side) to (0,side)
                p = 2 * side + (side - x)
            else:
                # left edge, from (0,side) down to (0,0)
                p = 3 * side + (side - y)
            pts.append((p, x, y))
        pts.sort(key=lambda t: t[0])
        # Extract sorted coords for speed
        px = [t[1] for t in pts]
        py = [t[2] for t in pts]

        # Check if we can pick k points with all pairwise Manhattan-dist >= D
        def feasible(D: int) -> bool:
            # We'll attempt up to 'k' different starting offsets
            # in the sorted list; if any yields k picks, return True.
            sel_x = []  # selected x-coords
            sel_y = []  # selected y-coords
            limit = n if n < k else k
            for start in range(limit):
                sel_x.clear()
                sel_y.clear()
                # pick the start point
                sel_x.append(px[start])
                sel_y.append(py[start])
                cnt = 1
                # greedily try to add more points
                for i in range(start+1, n):
                    vx = px[i]
                    vy = py[i]
                    ok = True
                    # check Manhattan distance to all already selected
                    for j in range(cnt):
                        dx = vx - sel_x[j]
                        if dx < 0: dx = -dx
                        dy = vy - sel_y[j]
                        if dy < 0: dy = -dy
                        if dx + dy < D:
                            ok = False
                            break
                    if ok:
                        sel_x.append(vx)
                        sel_y.append(vy)
                        cnt += 1
                        if cnt >= k:
                            return True
                # if this start fails, try next start
            return False

        # Binary search on answer D
        lo, hi = 0, 2 * side
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if feasible(mid):
                lo = mid
            else:
                hi = mid - 1
        return lo