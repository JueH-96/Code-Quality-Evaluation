class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        def is_reachable(x, y):
            if x < 0 or x > xCorner or y < 0 or y > yCorner:
                return False
            for circle in circles:
                if (x - circle[0])**2 + (y - circle[1])**2 <= circle[2]**2:
                    return False
            return True

        queue = [(0, 0)]
        visited = set()
        while queue:
            x, y = queue.pop(0)
            if (x, y) == (xCorner, yCorner):
                return True
            if (x, y) in visited:
                continue
            visited.add((x, y))
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                if is_reachable(x + dx, y + dy):
                    queue.append((x + dx, y + dy))
        return False