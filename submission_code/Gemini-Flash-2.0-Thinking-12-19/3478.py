import math

class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        n_circles = len(circles)
        parent = list(range(n_circles + 2))

        def find(i):
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]

        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j

        SOURCE = n_circles
        TARGET = n_circles + 1

        for i in range(n_circles):
            x_i, y_i, r_i = circles[i]
            if x_i - r_i <= 0 or y_i - r_i <= 0:
                union(i, SOURCE)
            if x_i + r_i >= xCorner or y_i + r_i >= yCorner:
                union(i, TARGET)

        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                if dist_sq <= (r1 + r2)**2:
                    union(i, j)

        return find(SOURCE) != find(TARGET)