class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        """
        Determines if a path exists from (0,0) to (xCorner, yCorner) within a rectangle,
        avoiding a set of circular obstacles.
        """

        # DSU (Union-Find) data structure to manage connected components.
        # It can be defined as a nested class or a helper class.
        class DSU:
            def __init__(self, n):
                self.parent = list(range(n))

            def find(self, i):
                if self.parent[i] == i:
                    return i
                # Path compression for efficiency
                self.parent[i] = self.find(self.parent[i])
                return self.parent[i]

            def union(self, i, j):
                root_i = self.find(i)
                root_j = self.find(j)
                if root_i != root_j:
                    self.parent[root_j] = root_i

        # Step 1: Filter for circles that actually intersect the rectangle.
        # A circle outside the rectangle can't block a path inside it.
        active_circles = []
        for x, y, r in circles:
            # Find the closest point in the rectangle to the circle's center.
            closest_x = max(0, min(x, xCorner))
            closest_y = max(0, min(y, yCorner))
            
            # If the distance from the center to this closest point is <= radius, they intersect.
            # Using squared distance to avoid floating-point math.
            dist_sq = (x - closest_x)**2 + (y - closest_y)**2
            if dist_sq <= r**2:
                active_circles.append([x, y, r])

        n_active = len(active_circles)
        
        # Step 2: Initialize DSU.
        # N nodes for active circles + 4 virtual nodes for the walls.
        dsu = DSU(n_active + 4)
        
        # Assign indices to virtual wall nodes for clarity.
        L_WALL, B_WALL, R_WALL, T_WALL = n_active, n_active + 1, n_active + 2, n_active + 3

        # Step 3: Build connectivity.
        # Connect circles to walls they touch.
        for i in range(n_active):
            x, y, r = active_circles[i]
            if x - r <= 0:
                dsu.union(i, L_WALL)
            if y - r <= 0:
                dsu.union(i, B_WALL)
            if x + r >= xCorner:
                dsu.union(i, R_WALL)
            if y + r >= yCorner:
                dsu.union(i, T_WALL)

        # Connect overlapping circles.
        for i in range(n_active):
            for j in range(i + 1, n_active):
                x1, y1, r1 = active_circles[i]
                x2, y2, r2 = active_circles[j]
                
                # Using squared distances for robustness with large numbers.
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                radii_sum_sq = (r1 + r2)**2
                
                if dist_sq <= radii_sum_sq:
                    dsu.union(i, j)
        
        # Step 4: Check for blocking conditions.
        # A path is blocked if there's a continuous chain of obstacles connecting:
        # - Left and Right walls (horizontal barrier)
        # - Bottom and Top walls (vertical barrier)
        # - Left and Bottom walls (trapping the start corner)
        # - Right and Top walls (trapping the end corner)
        
        if dsu.find(L_WALL) == dsu.find(R_WALL):
            return False
        if dsu.find(B_WALL) == dsu.find(T_WALL):
            return False
        if dsu.find(L_WALL) == dsu.find(B_WALL):
            return False
        if dsu.find(R_WALL) == dsu.find(T_WALL):
            return False
            
        return True