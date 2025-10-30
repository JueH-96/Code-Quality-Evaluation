from typing import List

class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        parent = list(range(len(circles)))
        rank = [0] * len(circles)
        
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u
        
        def union(u, v):
            u_root = find(u)
            v_root = find(v)
            if u_root == v_root:
                return
            if rank[u_root] < rank[v_root]:
                parent[u_root] = v_root
            else:
                parent[v_root] = u_root
                if rank[u_root] == rank[v_root]:
                    rank[u_root] += 1
        
        # Union circles that overlap
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dx = x1 - x2
                dy = y1 - y2
                if dx*dx + dy*dy <= (r1 + r2) * (r1 + r2):
                    union(i, j)
        
        # For each group, check if it connects left and right or top and bottom
        groups = {}
        for i in range(len(circles)):
            root = find(i)
            if root not in groups:
                groups[root] = {"left": False, "right": False, "top": False, "bottom": False}
            x, y, r = circles[i]
            if x - r <= 0:
                groups[root]["left"] = True
            if x + r >= xCorner:
                groups[root]["right"] = True
            if y - r <= 0:
                groups[root]["bottom"] = True
            if y + r >= yCorner:
                groups[root]["top"] = True
        
        for group in groups.values():
            if (group["left"] and group["right"]) or (group["top"] and group["bottom"]):
                return False
        return True