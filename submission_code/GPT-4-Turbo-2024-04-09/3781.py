from typing import List
from itertools import combinations

class Solution:
    def maxDistance(self, side: int, points: List[List[int]], k: int) -> int:
        def manhattan_distance(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
        # If k is equal to the number of points, the solution is to take all points
        if k == len(points):
            min_dist = float('inf')
            for p1, p2 in combinations(points, 2):
                min_dist = min(min_dist, manhattan_distance(p1, p2))
            return min_dist
        
        # Binary search for the maximum minimum distance
        def can_place_with_distance(d):
            # Try to place k points such that the minimum distance between any two is at least d
            # Using a greedy approach
            used_points = []
            for point in points:
                if not used_points:  # Always take the first point
                    used_points.append(point)
                else:
                    if all(manhattan_distance(point, used) >= d for used in used_points):
                        used_points.append(point)
                if len(used_points) == k:
                    return True
            return False
        
        # Binary search over possible distances
        low, high = 0, 2 * side
        while low < high:
            mid = (low + high + 1) // 2
            if can_place_with_distance(mid):
                low = mid
            else:
                high = mid - 1
        
        return low

# Example usage:
sol = Solution()
print(sol.maxDistance(2, [[0,2],[2,0],[2,2],[0,0]], 4))  # Output: 2
print(sol.maxDistance(2, [[0,0],[1,2],[2,0],[2,2],[2,1]], 4))  # Output: 1
print(sol.maxDistance(2, [[0,0],[0,1],[0,2],[1,2],[2,0],[2,2],[2,1]], 5))  # Output: 1