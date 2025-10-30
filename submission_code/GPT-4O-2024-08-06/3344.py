from typing import List

class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        def manhattan_distance(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
        n = len(points)
        
        # Calculate all Manhattan distances
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                distances.append((manhattan_distance(points[i], points[j]), i, j))
        
        # Sort distances in descending order
        distances.sort(reverse=True, key=lambda x: x[0])
        
        # Try removing each point and calculate the maximum distance
        min_max_distance = float('inf')
        for i in range(n):
            max_distance = 0
            for dist, p1, p2 in distances:
                if p1 != i and p2 != i:
                    max_distance = dist
                    break
            min_max_distance = min(min_max_distance, max_distance)
        
        return min_max_distance