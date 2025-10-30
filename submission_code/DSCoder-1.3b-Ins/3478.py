class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        def in_rectangle(x, y):
            return x <= xCorner and y <= yCorner

        def in_circle(x, y, r):
            return (x - r) ** 2 + (y - yCorner) ** 2 <= r ** 2

        def in_bounds(x, y):
            return 0 <= x <= xCorner and 0 <= y <= yCorner

        def dfs(x, y, visited):
            if (x, y) in visited:
                return False
            if in_rectangle(x, y):
                return True
            if in_circle(x, y, circles[0][2]):
                return False
            visited.add((x, y))
            return dfs(x + 1, y, visited) or dfs(x, y + 1, visited)

        return dfs(0, 0, set())