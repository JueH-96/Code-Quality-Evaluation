class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        # Check if start (0,0) is inside any circle
        for x, y, r in circles:
            if x * x + y * y <= r * r:
                return False
        
        # Check if end (xCorner, yCorner) is inside any circle
        for x, y, r in circles:
            dx = xCorner - x
            dy = yCorner - y
            if dx * dx + dy * dy <= r * r:
                return False
        
        # Filter circles that intersect the rectangle
        filtered = []
        for x, y, r in circles:
            dx = max(x - xCorner, -x, 0)
            dy = max(y - yCorner, -y, 0)
            dist_sq = dx * dx + dy * dy
            if dist_sq <= r * r:
                filtered.append((x, y, r))
        
        n = len(filtered)
        if n == 0:
            return True
        
        # Union-Find setup
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
                parent[pv] = pu  # Union by linking
        
        # Connect overlapping circles
        for i in range(n):
            xi, yi, ri = filtered[i]
            for j in range(i + 1, n):
                xj, yj, rj = filtered[j]
                dx = xi - xj
                dy = yi - yj
                dist_sq = dx * dx + dy * dy
                sum_r = ri + rj
                if dist_sq <= sum_r * sum_r:
                    union(i, j)
        
        # Check components for left-right or top-bottom connections
        component_info = {}
        for i in range(n):
            xi, yi, ri = filtered[i]
            left = (xi - ri) <= 0
            right = (xi + ri) >= xCorner
            top = (yi + ri) >= yCorner
            bottom = (yi - ri) <= 0
            root = find(i)
            if root not in component_info:
                component_info[root] = [left, right, top, bottom]
            else:
                component_info[root][0] |= left
                component_info[root][1] |= right
                component_info[root][2] |= top
                component_info[root][3] |= bottom
        
        for flags in component_info.values():
            left, right, top, bottom = flags
            if (left and right) or (top and bottom):
                return False
        
        return True