class DSU:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j
            return True
        return False

class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        num_circles = len(circles)

        # Check if start or end point is inside any circle
        for x, y, r in circles:
            if (0 - x)**2 + (0 - y)**2 < r**2:  # Strictly inside
                return False
            if (xCorner - x)**2 + (yCorner - y)**2 < r**2:  # Strictly inside
                return False

        dsu = DSU(num_circles + 2)  # Circles + start_boundary + end_boundary
        start_boundary_index = num_circles
        end_boundary_index = num_circles + 1

        # Connect overlapping circles
        for i in range(num_circles):
            for j in range(i + 1, num_circles):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                if dist_sq <= (r1 + r2)**2:
                    dsu.union(i, j)

        # Connect circles to boundaries
        for i in range(num_circles):
            x, y, r = circles[i]
            # Touches left or bottom
            if x - r <= 0 or y - r <= 0:
                dsu.union(i, start_boundary_index)
            # Touches right or top
            if x + r >= xCorner or y + r >= yCorner:
                dsu.union(i, end_boundary_index)

        # Check if start and end boundaries are connected
        return dsu.find(start_boundary_index) != dsu.find(end_boundary_index)