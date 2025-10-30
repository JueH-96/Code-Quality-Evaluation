from typing import List
import bisect

class Solution:
    def maxDistance(self, side: int, points: List[List[int]], k: int) -> int:
        # Map each boundary point (x, y) to a scalar position along the perimeter [0, 4*side)
        L = 4 * side
        pos = []
        for x, y in points:
            if x == 0 and 0 <= y <= side:
                s = y
            elif y == side and 0 <= x <= side:
                s = side + x
            elif x == side and 0 <= y <= side:
                s = 2 * side + (side - y)
            else:  # y == 0 and 0 <= x <= side
                s = 3 * side + (side - x)
            pos.append(s)
        pos.sort()
        n = len(pos)

        # Check if we can pick k points so that the minimal circular distance >= d
        def can(d: int) -> bool:
            # Try each point as the first pick
            for start in range(n):
                count = 1
                last = pos[start]
                # Greedily pick next points at least d apart (mod L)
                for _ in range(k - 1):
                    target = last + d
                    # find next in [target,∞), or wrap around to [0, target-L]
                    idx = bisect.bisect_left(pos, target)
                    if idx < n:
                        last = pos[idx]
                    else:
                        # wrap
                        wrap_target = target - L
                        idx2 = bisect.bisect_left(pos, wrap_target)
                        if idx2 < n:
                            last = pos[idx2]
                        else:
                            # no candidate
                            count = -1
                            break
                    count += 1
                    if count == k:
                        break
                if count == k:
                    # finally check wrap from last back to start
                    gap = (pos[start] + L - last)
                    if gap >= d:
                        return True
            return False

        # Binary search the largest d
        lo, hi = 0, L
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if can(mid):
                lo = mid
            else:
                hi = mid - 1
        return lo