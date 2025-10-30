from typing import List

class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        """
        We model every circle as an obstacle.  Two circles belong to one obstacle–component
        if they overlap or touch.  A component also “touches” a rectangle-side as soon as
        one of its circles touches that side.
        
        A path from (0,0) to (xCorner , yCorner) is impossible if one single obstacle–
        component produces one of the four “blocking joins” shown below

                 (1)  LEFT  ────── RIGHT        (horizontal wall)
                 (2)  TOP   ────── BOTTOM       (vertical   wall)
                 (3)  LEFT  ────── BOTTOM       (start corner locked in)
                 (4)  TOP   ────── RIGHT        (finish corner locked in)

        because in any of those situations the free interior of the rectangle
        is separated into at least two disconnected parts, one containing the start
        corner (0,0) and the other containing the destination (xCorner , yCorner).

        For every component we therefore only have to know which of the
        four sides {LEFT, RIGHT, TOP, BOTTOM} it touches.
        With that information we can decide immediately whether the component
        realizes one of the four “blocking joins” above.

        We keep the components by a union–find (disjoint-set) structure that
        contains a node for each circle and four extra nodes representing the
        rectangle sides.
        """

        # helper –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        class DSU:
            def __init__(self, n: int):
                self.par = list(range(n))
                self.rank = [0] * n
            def find(self, x: int) -> int:
                if self.par[x] != x:
                    self.par[x] = self.find(self.par[x])
                return self.par[x]
            def union(self, x: int, y: int):
                rx, ry = self.find(x), self.find(y)
                if rx == ry:
                    return
                if self.rank[rx] < self.rank[ry]:
                    rx, ry = ry, rx
                self.par[ry] = rx
                if self.rank[rx] == self.rank[ry]:
                    self.rank[rx] += 1
        # ---------------------------------------------------------------------

        n = len(circles)

        # first check that start/end are not inside any circle
        sx = sy = 0
        tx, ty = xCorner, yCorner
        for (cx, cy, r) in circles:
            r2 = r * r
            if (cx - sx) ** 2 + (cy - sy) ** 2 <= r2:
                return False
            if (cx - tx) ** 2 + (cy - ty) ** 2 <= r2:
                return False

        # indices of the 4 “side” nodes in the DSU
        LEFT, RIGHT, TOP, BOTTOM = 0, 1, 2, 3
        base = 4                          # first index that belongs to a circle
        dsu = DSU(base + n)

        # circle–side connections
        for idx, (cx, cy, r) in enumerate(circles):
            node = base + idx
            if cx - r <= 0:               # touches LEFT side
                dsu.union(node, LEFT)
            if cx + r >= xCorner:         # touches RIGHT side
                dsu.union(node, RIGHT)
            if cy - r <= 0:               # touches BOTTOM side
                dsu.union(node, BOTTOM)
            if cy + r >= yCorner:         # touches TOP side
                dsu.union(node, TOP)

        # circle–circle connections
        for i in range(n):
            cx1, cy1, r1 = circles[i]
            node1 = base + i
            for j in range(i + 1, n):
                cx2, cy2, r2 = circles[j]
                dx = cx1 - cx2
                dy = cy1 - cy2
                rr = r1 + r2
                if dx * dx + dy * dy <= rr * rr:      # overlap / touch
                    dsu.union(node1, base + j)

        # after everything is merged we only need to look whether
        # one component contains one of the four blocking side-pairs
        rootL = dsu.find(LEFT)
        rootR = dsu.find(RIGHT)
        rootT = dsu.find(TOP)
        rootB = dsu.find(BOTTOM)

        # (1) LEFT–RIGHT,  (2) TOP–BOTTOM,
        # (3) LEFT–BOTTOM, (4) TOP–RIGHT
        if rootL == rootR:
            return False
        if rootT == rootB:
            return False
        if rootL == rootB:
            return False
        if rootT == rootR:
            return False

        return True