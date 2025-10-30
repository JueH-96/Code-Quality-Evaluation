from typing import List

class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        def manhattan_distance(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

        n = len(points)
        max_distance = 0

        # Calculate the initial maximum distance without removing any point
        for i in range(n):
            for j in range(i + 1, n):
                max_distance = max(max_distance, manhattan_distance(points[i], points[j]))

        min_max_distance = float('inf')

        # Try removing each point and calculate the maximum distance for the remaining points
        for i in range(n):
            current_max_distance = 0
            for j in range(n):
                if i == j:
                    continue
                for k in range(j + 1, n):
                    if i == k:
                        continue
                    current_max_distance = max(current_max_distance, manhattan_distance(points[j], points[k]))
            min_max_distance = min(min_max_distance, current_max_distance)

        return min_max_distance