class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        n = len(points)
        points.sort()
        dp, dp2, dp3 = [0]*n, [0]*n, [0]*n
        for i in range(1, n):
            dp[i] = dp[i-1] if points[i][0] == points[i-1][0] else dp[i-1]+abs(points[i][0]-points[i-1][0])
            dp2[i] = min(dp2[i-1]+abs(points[i][0]-points[i-1][0]), dp[i-1]+abs(points[i][1]-points[i-1][1])) if points[i][1] >= points[i-1][1] else dp2[i-1]
            dp3[i] = min(dp3[i-1]+abs(points[i][0]-points[i-1][0]), dp2[i-1]+abs(points[i][1]-points[i-1][1])) if points[i][1] <= points[i-1][1] else dp3[i-1]
        return min(dp[-1], dp3[-1])