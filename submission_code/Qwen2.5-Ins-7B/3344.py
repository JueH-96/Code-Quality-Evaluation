from typing import List

class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        def manhattan_distance(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
        n = len(points)
        max_distances = [0] * n
        
        # Calculate max distance for each point when not removed
        for i in range(n):
            max_distance = 0
            for j in range(n):
                if i != j:
                    max_distance = max(max_distance, manhattan_distance(points[i], points[j]))
            max_distances[i] = max_distance
        
        # Find the minimum possible maximum distance after removing one point
        min_max_distance = float('inf')
        for i in range(n):
            min_max_distance = min(min_max_distance, max(max_distances[:i] + max_distances[i+1:]))
        
        return min_max_distance