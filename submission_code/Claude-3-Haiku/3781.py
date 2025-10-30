from typing import List
from collections import defaultdict

class Solution:
    def maxDistance(self, side: int, points: List[List[int]], k: int) -> int:
        # Sort the points based on their Manhattan distance from the origin (0, 0)
        points.sort(key=lambda x: abs(x[0]) + abs(x[1]))
        
        # Create a dictionary to group the points by their Manhattan distance from the origin
        distance_to_points = defaultdict(list)
        for x, y in points:
            distance = abs(x) + abs(y)
            distance_to_points[distance].append((x, y))
        
        # Iterate through the sorted distances and select the k points with the maximum minimum distance
        max_min_distance = 0
        selected_points = []
        for distance in sorted(distance_to_points.keys()):
            selected_points.extend(distance_to_points[distance])
            if len(selected_points) >= k:
                max_min_distance = distance
                break
        
        return max_min_distance