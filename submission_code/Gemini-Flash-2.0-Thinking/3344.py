from typing import List

class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        n = len(points)
        min_max_distance = float('inf')

        for i in range(n):
            remaining_points = []
            for j in range(n):
                if i != j:
                    remaining_points.append(points[j])

            if not remaining_points:
                continue

            max_dist_after_removal = 0
            num_remaining = len(remaining_points)
            for j in range(num_remaining):
                for k in range(j + 1, num_remaining):
                    p1 = remaining_points[j]
                    p2 = remaining_points[k]
                    dist = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
                    max_dist_after_removal = max(max_dist_after_removal, dist)

            min_max_distance = min(min_max_distance, max_dist_after_removal)

        return min_max_distance