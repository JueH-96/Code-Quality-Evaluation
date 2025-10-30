import random
from typing import List

class Solution:
    def maxDistance(self, side: int, points: List[List[int]], k: int) -> int:
        # Helper: can we pick k points with manh dist >= D?
        def can(D: int) -> bool:
            # Greedy check on current order
            def greedy_check(pts: List[List[int]]) -> bool:
                sel = []
                rem_need = k
                for x, y in pts:
                    # prune: if remaining points less than needed, stop
                    if len(pts) - idxs[0] < rem_need:
                        break
                    good = True
                    for sx, sy in sel:
                        if abs(sx - x) + abs(sy - y) < D:
                            good = False
                            break
                    if good:
                        sel.append((x, y))
                        rem_need -= 1
                        if rem_need == 0:
                            return True
                    idxs[0] += 1
                return False

            # Try a few random orderings
            pts = points
            # deterministic first try
            idxs[0] = 0
            if greedy_check(pts):
                return True
            # further random shuffles
            for _ in range(2):
                random.shuffle(pts)
                idxs[0] = 0
                if greedy_check(pts):
                    return True
            return False

        # Pre-allocate mutable index for pruning
        idxs = [0]
        # binary search D
        lo, hi = 0, 2 * side
        ans = 0
        # random seed for reproducibility
        random.seed(42)
        # Make a copy of points list
        pts = points[:]
        while lo <= hi:
            mid = (lo + hi) // 2
            # reset index counter
            idxs[0] = 0
            if can(mid):
                ans = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return ans