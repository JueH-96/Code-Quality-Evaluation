from typing import List

class DSU:
    def __init__(self, n):
        self.p = list(range(n))
    def find(self, x):
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]
    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.p[ry] = rx

class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        n = len(circles)
        # We add 4 special nodes: indices n..n+3 for LEFT, RIGHT, BOTTOM, TOP
        LEFT, RIGHT, BOTTOM, TOP = n, n+1, n+2, n+3
        dsu = DSU(n + 4)
        
        # Union circles that touch/overlap each other
        for i in range(n):
            xi, yi, ri = circles[i]
            # boundary checks
            if xi - ri <= 0:
                dsu.union(i, LEFT)
            if xi + ri >= xCorner:
                dsu.union(i, RIGHT)
            if yi - ri <= 0:
                dsu.union(i, BOTTOM)
            if yi + ri >= yCorner:
                dsu.union(i, TOP)
        
        # check circle-to-circle overlaps
        for i in range(n):
            xi, yi, ri = circles[i]
            for j in range(i+1, n):
                xj, yj, rj = circles[j]
                dx = xi - xj
                dy = yi - yj
                # if distance between centers <= sum of radii => they touch or overlap
                if dx*dx + dy*dy <= (ri + rj)*(ri + rj):
                    dsu.union(i, j)
        
        # If any single connected obstacle component connects:
        # LEFT & BOTTOM => blocks start corner
        # RIGHT & TOP   => blocks end corner
        # LEFT & RIGHT => full vertical barrier
        # BOTTOM & TOP => full horizontal barrier
        # then no valid path
        if dsu.find(LEFT) == dsu.find(BOTTOM):
            return False
        if dsu.find(RIGHT) == dsu.find(TOP):
            return False
        if dsu.find(LEFT) == dsu.find(RIGHT):
            return False
        if dsu.find(BOTTOM) == dsu.find(TOP):
            return False
        
        return True