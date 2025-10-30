import math

class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        min_max_dist = float('inf')
        n = len(points)
        for i in range(n):
            temp_points = []
            for j in range(n):
                if i != j:
                    temp_points.append(points[j])
            current_max_dist = 0
            num_remaining_points = len(temp_points)
            if num_remaining_points <= 1:
                current_max_dist = 0
            else:
                for j in range(num_remaining_points):
                    for k in range(j + 1, num_remaining_points):
                        p1 = temp_points[j]
                        p2 = temp_points[k]
                        dist = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
                        current_max_dist = max(current_max_dist, dist)
            min_max_dist = min(min_max_dist, current_max_dist)
        return min_max_dist