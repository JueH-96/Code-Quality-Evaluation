class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        n = len(points)
        min_max_dist = float('inf')

        for i in range(n):
            temp_points = points[:i] + points[i+1:]
            max_dist = 0
            for j in range(n - 1):
                for k in range(j + 1, n - 1):
                    dist = abs(temp_points[j][0] - temp_points[k][0]) + abs(temp_points[j][1] - temp_points[k][1])
                    max_dist = max(max_dist, dist)
            min_max_dist = min(min_max_dist, max_dist)

        return min_max_dist