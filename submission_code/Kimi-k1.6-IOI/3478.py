class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        # Check if start or end point is inside any circle
        for x, y, r in circles:
            dx1, dy1 = 0 - x, 0 - y
            if dx1 * dx1 + dy1 * dy1 <= r * r:
                return False
            dx2, dy2 = xCorner - x, yCorner - y
            if dx2 * dx2 + dy2 * dy2 <= r * r:
                return False
        
        n = len(circles)
        if n == 0:
            return True
        
        # Initialize DSU
        parent = list(range(n))
        
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]  # Path compression
                u = parent[u]
            return u
        
        def union(u, v):
            pu = find(u)
            pv = find(v)
            if pu != pv:
                parent[pu] = pv
        
        # Union circles that overlap
        for i in range(n):
            xi, yi, ri = circles[i]
            for j in range(i + 1, n):
                xj, yj, rj = circles[j]
                dx = xi - xj
                dy = yi - yj
                sum_r = ri + rj
                dist_sq = dx * dx + dy * dy
                if dist_sq <= sum_r * sum_r:
                    union(i, j)
        
        # Check each component for barriers
        components = {}
        for i in range(n):
            root = find(i)
            if root not in components:
                components[root] = {'left': False, 'right': False, 'bottom': False, 'top': False}
            x, y, r = circles[i]
            # Check edges
            left = (x - r <= 0) and (x + r >= 0)
            right = (x - r <= xCorner) and (x + r >= xCorner)
            bottom = (y - r <= 0) and (y + r >= 0)
            top = (y - r <= yCorner) and (y + r >= yCorner)
            
            if left:
                components[root]['left'] = True
            if right:
                components[root]['right'] = True
            if bottom:
                components[root]['bottom'] = True
            if top:
                components[root]['top'] = True
        
        # Determine if any component forms a barrier
        for comp in components.values():
            if (comp['left'] and comp['right']) or (comp['bottom'] and comp['top']):
                return False
        
        return True