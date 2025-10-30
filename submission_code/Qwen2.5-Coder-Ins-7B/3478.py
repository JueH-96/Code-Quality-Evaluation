class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        def is_valid(x, y):
            for cx, cy, r in circles:
                if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                    return False
            return True
        
        def dfs(x, y):
            if x == xCorner and y == yCorner:
                return True
            if x > xCorner or y > yCorner:
                return False
            if not is_valid(x, y):
                return False
            if (x, y) in visited:
                return False
            visited.add((x, y))
            return dfs(x + 1, y) or dfs(x, y + 1)
        
        visited = set()
        return dfs(0, 0)