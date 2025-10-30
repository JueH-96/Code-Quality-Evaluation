class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        def get_distance(p1, p2):
            x1, y1 = divmod(p1, 60)
            x2, y2 = divmod(p2, 60)
            return abs(x1 - x2) + abs(y1 - y2)

        def dp(i, f):
            if i == len(points):
                return 0 if f else float('inf')
            if (i, f) in dp.cache:
                return dp.cache[(i, f)]
            if f:
                dp.cache[(i, f)] = min(dp(i + 1, 0) + get_distance(points[i][0], 100), dp(i + 1, 1) + get_distance(100, points[i][1]))
            else:
                dp.cache[(i, f)] = min(dp(i + 1, 0) + get_distance(points[i][0], 100), dp(i + 1, 1) + get_distance(100, points[i][1]))
            return dp.cache[(i, f)]

        dp.cache = {}
        return dp(0, 0)