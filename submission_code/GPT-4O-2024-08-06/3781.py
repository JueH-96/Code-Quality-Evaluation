from typing import List
from itertools import combinations

class Solution:
    def maxDistance(self, side: int, points: List[List[int]], k: int) -> int:
        def manhattan_dist(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
        def is_valid(d):
            # Try to select k points such that all pairwise distances are at least d
            selected = [points[0]]
            for point in points[1:]:
                if len(selected) == k:
                    break
                if all(manhattan_dist(point, sel) >= d for sel in selected):
                    selected.append(point)
            return len(selected) == k
        
        # Sort points to have a deterministic order
        points.sort()
        
        # Binary search for the maximum minimum distance
        low, high = 0, 2 * side
        while low < high:
            mid = (low + high + 1) // 2
            if is_valid(mid):
                low = mid
            else:
                high = mid - 1
        
        return low