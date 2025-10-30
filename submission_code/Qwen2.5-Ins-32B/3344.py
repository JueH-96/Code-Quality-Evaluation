from typing import List

class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        n = len(points)
        max_distances = [0] * n
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])
                    max_distances[i] = max(max_distances[i], dist)
        
        return min(max_distances)