class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        x = sorted(points, key=lambda p: p[0] - p[1])
        y = sorted(points, key=lambda p: p[0] + p[1])
        res = float('inf')
        for a, b in ((x, y), (y, x)):
            d = [a[-1][0] - a[0][0], a[-1][1] - a[0][1]]
            for i in range(-2, 0):
                res = min(res, max(abs(a[i][0] - a[0][0]), abs(a[i][1] - a[0][1])) + max(abs(b[-1][0] - b[0][0]), abs(b[-1][1] - b[0][1])))
                res = min(res, max(abs(a[-1][0] - a[i][0]), abs(a[-1][1] - a[i][1])) + max(abs(b[-1][0] - b[0][0]), abs(b[-1][1] - b[0][1])))
        return res