from typing import List

class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        def manhattan_distance(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
        def max_distance(points):
            max_dist = 0
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    max_dist = max(max_dist, manhattan_distance(points[i], points[j]))
            return max_dist
        
        min_max_distance = float('inf')
        for i in range(len(points)):
            new_points = points[:i] + points[i+1:]
            max_dist = max_distance(new_points)
            min_max_distance = min(min_max_distance, max_dist)
        
        return min_max_distance