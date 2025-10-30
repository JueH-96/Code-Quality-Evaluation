from typing import List

class DSU:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0]*n

    def find(self, x):
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.r[rx] < self.r[ry]:
            self.p[rx] = ry
        else:
            self.p[ry] = rx
            if self.r[rx] == self.r[ry]:
                self.r[rx] += 1

class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        n = len(circles)
        # indices for walls
        LEFT, RIGHT, BOTTOM, TOP = n, n+1, n+2, n+3
        dsu = DSU(n+4)
        
        # helper to check wall-segment intersection
        for i, (x, y, r) in enumerate(circles):
            r2 = r*r
            # left wall x=0
            dx = x
            if dx <= r:
                # compute vertical interval intersection
                # if center y <= yCorner, it automatically overlaps segment
                if y <= yCorner:
                    dsu.union(i, LEFT)
                else:
                    # y > yCorner -> need (r^2 - dx^2) >= (y - yCorner)^2
                    if r2 - dx*dx >= (y - yCorner)*(y - yCorner):
                        dsu.union(i, LEFT)
            # right wall x = xCorner
            dx = abs(x - xCorner)
            if dx <= r:
                if y <= yCorner:
                    dsu.union(i, RIGHT)
                else:
                    if r2 - dx*dx >= (y - yCorner)*(y - yCorner):
                        dsu.union(i, RIGHT)
            # bottom wall y=0
            dy = y
            if dy <= r:
                if x <= xCorner:
                    dsu.union(i, BOTTOM)
                else:
                    if r2 - dy*dy >= (x - xCorner)*(x - xCorner):
                        dsu.union(i, BOTTOM)
            # top wall y = yCorner
            dy = abs(y - yCorner)
            if dy <= r:
                if x <= xCorner:
                    dsu.union(i, TOP)
                else:
                    if r2 - dy*dy >= (x - xCorner)*(x - xCorner):
                        dsu.union(i, TOP)
        
        # union intersecting circles
        for i in range(n):
            x1, y1, r1 = circles[i]
            for j in range(i+1, n):
                x2, y2, r2 = circles[j]
                dx = x1 - x2
                dy = y1 - y2
                rr = r1 + r2
                if dx*dx + dy*dy <= rr*rr:
                    dsu.union(i, j)
        
        # check blocking configurations
        # 1) barrier left-right (blocks top/bottom separation)
        if dsu.find(LEFT) == dsu.find(RIGHT):
            return False
        # 2) barrier top-bottom (blocks left/right separation)
        if dsu.find(TOP) == dsu.find(BOTTOM):
            return False
        # 3) barrier left-bottom isolates start corner
        if dsu.find(LEFT) == dsu.find(BOTTOM):
            return False
        # 4) barrier top-right isolates end corner
        if dsu.find(TOP) == dsu.find(RIGHT):
            return False
        
        # if none of the blocking barriers exist, path exists
        return True