import math
# The List type hint requires importing from typing module
from typing import List 

# Disjoint Set Union (DSU) implementation
class DSU:
    """
    A simple Disjoint Set Union (DSU) data structure implementation.
    It supports find operation with path compression for efficiency.
    """
    def __init__(self, n: int):
        """Initializes the DSU structure for n elements (0 to n-1)."""
        # Each element starts in its own set, represented by its own index.
        self.parent = list(range(n))
    
    def find(self, i: int) -> int:
        """
        Finds the representative (root) of the set containing element i.
        Applies path compression heuristic: makes elements along the path point directly to the root.
        """
        # Base case: if the element is its own parent, it's the root.
        if self.parent[i] == i:
            return i
        # Recursively find the root of the parent element.
        root = self.find(self.parent[i])
        # Path compression: update the parent of i to point directly to the root.
        self.parent[i] = root 
        return root
        
    def union(self, i: int, j: int) -> bool:
        """
        Unites the sets containing elements i and j.
        If i and j are already in the same set, does nothing.
        Returns True if a union was performed (i and j were in different sets), False otherwise.
        """
        # Find the roots of the sets containing i and j.
        root_i = self.find(i)
        root_j = self.find(j)
        
        # If they are already in the same set, no union needed.
        if root_i != root_j:
            # Union strategy: make the root of i's set point to the root of j's set.
            # Other strategies like union by rank/size exist but are not strictly necessary here.
            self.parent[root_i] = root_j
            return True # A merge happened.
        return False # They were already in the same set.

class Solution:
    """ 
    Solution class for the problem of checking path existence in a rectangle avoiding circles.
    """
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        """
        Determines if a path exists from (0,0) to (xCorner, yCorner) within a rectangle defined
        by these corners. The path must stay strictly inside the rectangle (except at the endpoints)
        and must not touch or enter any of the given circles.

        Args:
            xCorner: The x-coordinate of the top-right corner of the rectangle.
            yCorner: The y-coordinate of the top-right corner of the rectangle.
            circles: A list of circles, where each circle is represented as [x_center, y_center, radius].

        Returns:
            True if such a path exists, False otherwise.
            
        Approach:
        Models the problem using connectivity analysis with Disjoint Set Union (DSU).
        Each circle is treated as a node. Four additional virtual nodes represent the boundaries:
        Left (x=0), Right (x=xCorner), Bottom (y=0), Top (y=yCorner).
        
        Connectivity rules:
        1. Two circles are connected if they overlap or touch.
        2. A circle is connected to a boundary node if it intersects or touches the corresponding boundary line.
        
        Blocking conditions:
        A path from (0,0) to (xCorner, yCorner) is blocked if there exists a connected component of circles that:
        - Spans from the Left boundary to the Right boundary.
        OR
        - Spans from the Bottom boundary to the Top boundary.
        
        If either of these blocking conditions is met, return False. Otherwise, return True.
        """
        N = len(circles)
        
        # Initialize DSU with N nodes for circles + 4 virtual nodes for boundaries.
        # Indices 0..N-1: circles.
        # Indices N, N+1, N+2, N+3: Left, Right, Bottom, Top boundaries respectively.
        dsu = DSU(N + 4)
        
        # Define indices for virtual boundary nodes for clarity and maintainability.
        LEFT_BOUNDARY_NODE = N
        RIGHT_BOUNDARY_NODE = N + 1
        BOTTOM_BOUNDARY_NODE = N + 2
        TOP_BOUNDARY_NODE = N + 3
        
        # Process all circles to establish connectivity.
        for i in range(N):
            x1, y1, r1 = circles[i]
            
            # Check for overlaps with other circles (j > i to check each pair once).
            for j in range(i + 1, N):
                x2, y2, r2 = circles[j]
                
                # Calculate squared distance between circle centers.
                # Use Python's arbitrary precision integers to avoid overflow with large coordinates.
                dist_x = x1 - x2
                dist_y = y1 - y2
                dist_sq = dist_x**2 + dist_y**2
                
                # Calculate the sum of radii and its square.
                sum_radii = r1 + r2
                sum_radii_sq = sum_radii**2 
                
                # If distance squared <= squared sum of radii, circles overlap or touch. Union their sets.
                if dist_sq <= sum_radii_sq:
                    dsu.union(i, j)

            # Check intersections of circle 'i' with the four boundaries.
            # A circle intersects a boundary line if the distance from its center to the line is less than or equal to its radius.

            # Check Left boundary (line x=0). Distance is x1 (since x1 >= 1 is given implicitly by constraints, |x1|=x1).
            if x1 <= r1:
                dsu.union(i, LEFT_BOUNDARY_NODE)
            
            # Check Right boundary (line x=xCorner). Distance is |x1 - xCorner|.
            if abs(x1 - xCorner) <= r1:
                 dsu.union(i, RIGHT_BOUNDARY_NODE)
                
            # Check Bottom boundary (line y=0). Distance is y1 (since y1 >= 1).
            if y1 <= r1:
                dsu.union(i, BOTTOM_BOUNDARY_NODE)
                
            # Check Top boundary (line y=yCorner). Distance is |y1 - yCorner|.
            if abs(y1 - yCorner) <= r1:
                dsu.union(i, TOP_BOUNDARY_NODE)
                
        # After processing all circles and their connections, check the blocking conditions.
        
        # Check if Left boundary node and Right boundary node are in the same connected component.
        # Use dsu.find() to get the representative (root) of each node's set.
        if dsu.find(LEFT_BOUNDARY_NODE) == dsu.find(RIGHT_BOUNDARY_NODE):
            # If connected, there's a barrier blocking vertical movement across the rectangle.
            return False 
            
        # Check if Bottom boundary node and Top boundary node are in the same connected component.
        if dsu.find(BOTTOM_BOUNDARY_NODE) == dsu.find(TOP_BOUNDARY_NODE):
            # If connected, there's a barrier blocking horizontal movement across the rectangle.
            return False 
            
        # If neither blocking condition is met, it implies no complete barrier exists,
        # allowing a path from (0,0) to (xCorner, yCorner).
        return True