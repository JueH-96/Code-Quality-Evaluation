from typing import List

class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        def manhattan_distance(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

        n = len(points)
        if n == 3:
            return max(manhattan_distance(points[0], points[1]),
                       manhattan_distance(points[1], points[2]),
                       manhattan_distance(points[0], points[2]))

        min_max_distance = float('inf')

        for i in range(n):
            max_distance = 0
            for j in range(n):
                if i != j:
                    max_distance = max(max_distance, manhattan_distance(points[i], points[j]))
            min_max_distance = min(min_max_distance, max_distance)

        return min_max_distance