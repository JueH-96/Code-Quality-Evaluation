from typing import List

class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        # Quick check: start or end lies inside or on any circle -> no valid path
        for x, y, r in circles:
            # start = (0,0)
            if x*x + y*y <= r*r:
                return False
            # end = (xCorner, yCorner)
            dx = x - xCorner
            dy = y - yCorner
            if dx*dx + dy*dy <= r*r:
                return False

        n = len(circles)
        # Union-Find setup
        parent = list(range(n))
        rank = [0]*n

        def find(i: int) -> int:
            # path compression
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            ri = find(i)
            rj = find(j)
            if ri == rj:
                return
            # union by rank
            if rank[ri] < rank[rj]:
                parent[ri] = rj
            elif rank[ri] > rank[rj]:
                parent[rj] = ri
            else:
                parent[rj] = ri
                rank[ri] += 1

        # Precompute which walls each circle touches
        # We will track per-circle flags, then merge per-component.
        # flags[i] = (touches_left, touches_right, touches_bottom, touches_top)
        flags = []
        for (x, y, r) in circles:
            tl = (x <= r)            # touches left wall x=0
            tr = (x + r >= xCorner)  # touches right wall x=xCorner
            tb = (y <= r)            # touches bottom wall y=0
            tt = (y + r >= yCorner)  # touches top wall y=yCorner
            flags.append([tl, tr, tb, tt])

        # Union circles that overlap or touch
        # Two circles i,j connect if distance_centers <= r_i + r_j
        for i in range(n):
            xi, yi, ri = circles[i]
            for j in range(i+1, n):
                xj, yj, rj = circles[j]
                dx = xi - xj
                dy = yi - yj
                # compare squared distances
                if dx*dx + dy*dy <= (ri + rj)*(ri + rj):
                    union(i, j)

        # For each component, OR together the wall-touch flags
        comp_flags = {}  # root -> [tl, tr, tb, tt]
        for i in range(n):
            r_i = find(i)
            if r_i not in comp_flags:
                comp_flags[r_i] = flags[i].copy()
            else:
                # merge flags
                f = comp_flags[r_i]
                fi = flags[i]
                f[0] = f[0] or fi[0]
                f[1] = f[1] or fi[1]
                f[2] = f[2] or fi[2]
                f[3] = f[3] or fi[3]

        # Determine if there's a barrier on both diagonal "arcs":
        # Arc1 = bottom & right  => blocks the "upper-left" detour
        # Arc2 = left & top     => blocks the "bottom-right" detour
        blockBR = False  # bottom+right barrier exists
        blockLT = False  # left+top barrier exists
        for (tl, tr, tb, tt) in comp_flags.values():
            if tb and tr:
                blockBR = True
            if tl and tt:
                blockLT = True
            # early exit if both barriers present
            if blockBR and blockLT:
                return False

        # If both barrier types are present, no path; otherwise there is a way
        return True