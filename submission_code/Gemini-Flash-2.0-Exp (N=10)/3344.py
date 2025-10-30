from typing import List

class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        n = len(points)
        min_max_dist = float('inf')

        for k in range(n):
            temp_points = points[:k] + points[k+1:]
            max_dist = 0
            for i in range(len(temp_points)):
                for j in range(i + 1, len(temp_points)):
                    dist = abs(temp_points[i][0] - temp_points[j][0]) + abs(temp_points[i][1] - temp_points[j][1])
                    max_dist = max(max_dist, dist)
            min_max_dist = min(min_max_dist, max_dist)
        
        return min_max_dist