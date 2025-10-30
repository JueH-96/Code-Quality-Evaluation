from typing import List

class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        def manhattan_distance(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
        # Calculate the maximum distances from each point to all other points
        max_distances = [0] * len(points)
        for i in range(len(points)):
            for j in range(len(points)):
                if i != j:
                    dist = manhattan_distance(points[i], points[j])
                    max_distances[i] = max(max_distances[i], dist)
        
        # Find the minimum of the maximum distances after removing each point
        min_max_distance = float('inf')
        for i in range(len(points)):
            # Calculate the new maximum distance after removing point i
            new_max_distance = 0
            for j in range(len(points)):
                if i != j:
                    for k in range(j + 1, len(points)):
                        if i != k:
                            dist = manhattan_distance(points[j], points[k])
                            new_max_distance = max(new_max_distance, dist)
            min_max_distance = min(min_max_distance, new_max_distance)
        
        return min_max_distance

# Example usage:
# sol = Solution()
# print(sol.minimumDistance([[3,10],[5,15],[10,2],[4,4]]))  # Output: 12
# print(sol.minimumDistance([[1,1],[1,1],[1,1]]))  # Output: 0