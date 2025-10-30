class DSU:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root == y_root:
            return
        if self.rank[x_root] < self.rank[y_root]:
            self.parent[x_root] = y_root
        else:
            self.parent[y_root] = x_root
            if self.rank[x_root] == self.rank[y_root]:
                self.rank[x_root] += 1

class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        n = len(circles)
        if n == 0:
            return True
        dsu = DSU(4 + n)
        LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 3
        
        for i in range(n):
            x, y, r = circles[i]
            touches = []
            if x - r <= 0:
                touches.append(LEFT)
            if x + r >= xCorner:
                touches.append(RIGHT)
            if y - r <= 0:
                touches.append(BOTTOM)
            if y + r >= yCorner:
                touches.append(TOP)
            for edge in touches:
                dsu.union(4 + i, edge)
        
        for i in range(n):
            x1, y1, r1 = circles[i]
            for j in range(i + 1, n):
                x2, y2, r2 = circles[j]
                dx = x1 - x2
                dy = y1 - y2
                dist_sq = dx * dx + dy * dy
                if dist_sq <= (r1 + r2) ** 2:
                    dsu.union(4 + i, 4 + j)
        
        return not (dsu.find(LEFT) == dsu.find(RIGHT) or dsu.find(BOTTOM) == dsu.find(TOP))