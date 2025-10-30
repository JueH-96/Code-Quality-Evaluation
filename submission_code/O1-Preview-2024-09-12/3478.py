class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        n = len(circles)
        parent = [i for i in range(n + 2)]  # extra two nodes for borders
        rank = [0] * (n + 2)
        
        LEFT_BOTTOM = n      # index for left/bottom border node
        RIGHT_TOP = n + 1    # index for right/top border node
        
        # Helper function for Union-Find
        def find(u):
            if parent[u] != u:
                parent[u] = find(parent[u])
            return parent[u]
        
        def union(u, v):
            pu, pv = find(u), find(v)
            if pu == pv:
                return
            if rank[pu] < rank[pv]:
                parent[pu] = pv
            else:
                parent[pv] = pu
                if rank[pu] == rank[pv]:
                    rank[pu] += 1
        
        # Check if starting or ending corner lies inside any circle
        for i, (x, y, r) in enumerate(circles):
            # Check (0, 0)
            if (x ** 2 + y ** 2) <= r ** 2:
                return False
            # Check (xCorner, yCorner)
            if ((x - xCorner) ** 2 + (y - yCorner) ** 2) <= r ** 2:
                return False
        
        # For each circle, check connections to borders
        for i, (x, y, r) in enumerate(circles):
            # Left or Bottom border
            if x - r <= 0 or y - r <= 0:
                union(i, LEFT_BOTTOM)
            # Right or Top border
            if x + r >= xCorner or y + r >= yCorner:
                union(i, RIGHT_TOP)
        
        # For each pair of circles, check if they touch
        for i in range(n):
            x1, y1, r1 = circles[i]
            for j in range(i + 1, n):
                x2, y2, r2 = circles[j]
                dx = x1 - x2
                dy = y1 - y2
                dist_sq = dx * dx + dy * dy
                radius_sum = r1 + r2
                if dist_sq <= radius_sum ** 2:
                    union(i, j)
        
        # Check if left/bottom border and right/top border are connected
        if find(LEFT_BOTTOM) == find(RIGHT_TOP):
            return False
        else:
            return True