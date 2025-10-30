import math
from typing import List

# DSU (Disjoint Set Union) class
# This class helps manage connected components.
class DSU:
    def __init__(self, n: int):
        # parent[i] stores the parent of element i.
        # Initially, each element is its own parent.
        self.parent = list(range(n))
        
    def find(self, i: int) -> int:
        # Find the representative (root) of the set containing i.
        # Uses path compression for optimization.
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int) -> bool:
        # Unites the sets containing i and j.
        # Returns True if a union occurred (i.e., i and j were in different sets), False otherwise.
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j
            # Union by rank/size optimization could be added here for theoretical improvements,
            # but for N=1000, it's not strictly necessary.
            return True
        return False

class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        N = len(circles)
        
        # Define indices for the DSU nodes:
        # 0 to N-1: represent the circles
        # N: represents the Left boundary (x=0)
        # N+1: represents the Right boundary (x=xCorner)
        # N+2: represents the Bottom boundary (y=0)
        # N+3: represents the Top boundary (y=yCorner)
        LEFT_BOUNDARY_IDX = N
        RIGHT_BOUNDARY_IDX = N + 1
        BOTTOM_BOUNDARY_IDX = N + 2
        TOP_BOUNDARY_IDX = N + 3
        
        # Initialize DSU with N circles + 4 boundary nodes
        dsu = DSU(N + 4)

        # Step 1: Pre-check if the start (0,0) or end (xCorner, yCorner) points
        # are inside or touching any circle. If so, no path is possible.
        for x_c, y_c, r_c in circles:
            # Check (0,0)
            # Squared distance from (0,0) to (x_c, y_c)
            dist_sq_from_origin = x_c**2 + y_c**2
            if dist_sq_from_origin <= r_c**2:
                return False # (0,0) is blocked
            
            # Check (xCorner, yCorner)
            # Squared distance from (xCorner, yCorner) to (x_c, y_c)
            dist_sq_from_corner = (x_c - xCorner)**2 + (y_c - yCorner)**2
            if dist_sq_from_corner <= r_c**2:
                return False # (xCorner, yCorner) is blocked

        # Step 2: Connect circles to rectangle boundaries if they touch or overlap
        for i in range(N):
            x, y, r = circles[i]
            
            # Connect to Left boundary (x=0) if circle touches or crosses it
            if x - r <= 0:
                dsu.union(i, LEFT_BOUNDARY_IDX)
            
            # Connect to Right boundary (x=xCorner) if circle touches or crosses it
            if x + r >= xCorner:
                dsu.union(i, RIGHT_BOUNDARY_IDX)
            
            # Connect to Bottom boundary (y=0) if circle touches or crosses it
            if y - r <= 0:
                dsu.union(i, BOTTOM_BOUNDARY_IDX)
            
            # Connect to Top boundary (y=yCorner) if circle touches or crosses it
            if y + r >= yCorner:
                dsu.union(i, TOP_BOUNDARY_IDX)

        # Step 3: Connect overlapping or touching circles to each other
        for i in range(N):
            x1, y1, r1 = circles[i]
            for j in range(i + 1, N): # Iterate through unique pairs
                x2, y2, r2 = circles[j]
                
                # Calculate squared Euclidean distance between centers
                squared_dist_centers = (x1 - x2)**2 + (y1 - y2)**2
                
                # Calculate squared sum of radii
                sum_radii = r1 + r2
                squared_sum_radii = sum_radii**2
                
                # If squared distance <= squared sum of radii, circles overlap or touch
                if squared_dist_centers <= squared_sum_radii:
                    dsu.union(i, j)
        
        # Step 4: Determine if a path exists based on connected components of boundaries
        # A path exists if there is NO barrier spanning horizontally AND NO barrier spanning vertically.
        
        # Check for horizontal barrier (Left boundary connected to Right boundary)
        left_to_right_blocked = (dsu.find(LEFT_BOUNDARY_IDX) == dsu.find(RIGHT_BOUNDARY_IDX))
        
        # Check for vertical barrier (Bottom boundary connected to Top boundary)
        bottom_to_top_blocked = (dsu.find(BOTTOM_BOUNDARY_IDX) == dsu.find(TOP_BOUNDARY_IDX))
        
        # A path is blocked if either the horizontal or vertical passage is blocked.
        # So, a path exists if and only if NOT (horizontal barrier OR vertical barrier).
        return not (left_to_right_blocked or bottom_to_top_blocked)