from typing import List

# Define the DSU class outside the Solution class for clarity
class DSU:
    def __init__(self, n):
        # Initialize parent array: each node is its own parent initially
        self.parent = list(range(n))
        # Initialize rank (or size) array: used for optimizing union by rank/size
        self.rank = [0] * n

    def find(self, i):
        # Find the representative (root) of the set containing element i
        if self.parent[i] == i:
            # If i is the parent of itself, it is the root
            return i
        # Path compression: set the parent of i directly to the root
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        # Unite the sets containing elements i and j
        root_i = self.find(i)
        root_j = self.find(j)
        
        # If i and j are already in the same set, do nothing
        if root_i != root_j:
            # Union by rank: attach the smaller tree under the root of the larger tree
            # (or based on rank, which is an upper bound on tree height)
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                # If ranks are equal, choose one as root and increment its rank
                self.parent[root_j] = root_i
                self.rank[root_i] += 1
            # Return True indicating sets were merged
            return True
        # Return False indicating sets were already the same
        return False

class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        # Get the number of circles
        n = len(circles)
        
        # Create a DSU structure. Nodes represent circles (0 to n-1) and
        # the four edges of the rectangle (n to n+3).
        # n: bottom edge (y=0)
        # n+1: left edge (x=0)
        # n+2: top edge (y=yCorner)
        # n+3: right edge (x=xCorner)
        dsu = DSU(n + 4)

        # Assign intuitive names to the boundary node indices
        bottom_edge_node = n
        left_edge_node = n + 1
        top_edge_node = n + 2
        right_edge_node = n + 3

        # Check if any circle covers the start corner (0, 0) or the end corner (xCorner, yCorner).
        # If a corner is covered, no path can start or end there.
        for i in range(n):
            xi, yi, ri = circles[i]
            
            # Calculate squared distance from circle center (xi, yi) to (0, 0)
            # Since xi, yi are positive, dist_sq = xi^2 + yi^2
            dist_sq_start = xi * xi + yi * yi
            # Calculate squared radius
            ri_sq = ri * ri
            
            # If distance squared <= radius squared, the circle covers (0, 0)
            if dist_sq_start <= ri_sq:
                return False

            # Check end corner (xCorner, yCorner)
            # dist_sq = (xi - xCorner)^2 + (yi - yCorner)^2
            dist_sq_end = (xi - xCorner) * (xi - xCorner) + (yi - yCorner) * (yi - yCorner)
            
            # If distance squared <= radius squared, the circle covers (xCorner, yCorner)
            if dist_sq_end <= ri_sq:
                return False

        # Connect circles to each other if they touch or overlap.
        # A path cannot pass between two circles if the distance between their centers
        # is less than or equal to the sum of their radii.
        for i in range(n):
            for j in range(i + 1, n): # Check each pair of distinct circles
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                
                # Calculate distance squared between centers (xi, yi) and (xj, yj)
                dist_sq = (xi - xj) * (xi - xj) + (yi - yj) * (yi - yj)
                
                # Calculate sum of radii and its square
                sum_radii = ri + rj
                sum_radii_sq = sum_radii * sum_radii
                
                # If distance squared <= sum of radii squared, the circles touch or overlap
                if dist_sq <= sum_radii_sq:
                    # Union the sets containing circle i and circle j
                    dsu.union(i, j)

        # Connect circles to the rectangle boundaries if they touch or overlap.
        # A circle touching a boundary edge acts as an obstacle connected to that edge.
        for i in range(n):
            xi, yi, ri = circles[i]
            
            # Check bottom edge (y=0)
            # A circle touches or overlaps y=0 if its center's y-coordinate is within radius distance
            # of 0. Since yi >= 1 and ri >= 1, this simplifies to yi <= ri.
            if yi <= ri:
                # Union the set containing circle i with the set representing the bottom edge
                dsu.union(i, bottom_edge_node)
                
            # Check left edge (x=0)
            # Similar to the bottom edge. Since xi >= 1 and ri >= 1, this is xi <= ri.
            if xi <= ri:
                # Union the set containing circle i with the set representing the left edge
                dsu.union(i, left_edge_node)
                
            # Check top edge (y=yCorner)
            # A circle touches or overlaps y=yCorner if its center's y-coordinate is within radius distance
            # of yCorner. Since yi >= 1, ri >= 1, and yCorner >= 3, this condition is abs(yi - yCorner) <= ri.
            # The check yi + ri >= yCorner is equivalent and simpler given the positive coordinates.
            if yi + ri >= yCorner:
                # Union the set containing circle i with the set representing the top edge
                dsu.union(i, top_edge_node)
                
            # Check right edge (x=xCorner)
            # Similar to the top edge. Since xi >= 1, ri >= 1, and xCorner >= 3,
            # the check xi + ri >= xCorner is used.
            if xi + ri >= xCorner:
                # Union the set containing circle i with the set representing the right edge
                dsu.union(i, right_edge_node)

        # A path exists from (0,0) to (xCorner, yCorner) within the rectangle, avoiding circles,
        # if and only if there is no continuous barrier of circles and/or boundaries
        # separating the start region (near bottom/left edges) from the end region (near top/right edges).
        # The critical barriers are those connecting the bottom edge to the top edge,
        # OR connecting the left edge to the right edge.
        
        # Check if the bottom edge is connected to the top edge through the circles/boundaries graph
        if dsu.find(bottom_edge_node) == dsu.find(top_edge_node):
            # If they are in the same set, a barrier exists
            return False

        # Check if the left edge is connected to the right edge through the circles/boundaries graph
        if dsu.find(left_edge_node) == dsu.find(right_edge_node):
            # If they are in the same set, a barrier exists
            return False

        # If neither barrier exists, a path is possible
        return True