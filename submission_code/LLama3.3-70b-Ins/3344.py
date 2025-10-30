from typing import List

class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        def manhattan_distance(point1, point2):
            """Calculate the Manhattan distance between two points."""
            return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

        def max_distance(points):
            """Calculate the maximum Manhattan distance between any two points."""
            max_dist = 0
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    max_dist = max(max_dist, manhattan_distance(points[i], points[j]))
            return max_dist

        min_max_dist = float('inf')
        for i in range(len(points)):
            # Create a new list of points by removing the i-th point
            new_points = points[:i] + points[i + 1:]
            # Calculate the maximum distance for the new list of points
            max_dist = max_distance(new_points)
            # Update the minimum maximum distance
            min_max_dist = min(min_max_dist, max_dist)

        return min_max_dist