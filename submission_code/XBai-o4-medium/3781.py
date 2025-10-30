from typing import List

class Solution:
    def maxDistance(self, side: int, points: List[List[int]], k: int) -> int:
        def perimeter_distance(point):
            x, y = point
            if y == 0:
                return x
            elif x == side:
                return side + y
            elif y == side:
                return 2 * side + (side - x)
            elif x == 0:
                return 3 * side + (side - y)
            # This else case is theoretically unreachable since points are on the square boundary
            return 0  # just to satisfy the compiler, though not needed per problem constraints
        
        # Sort the points based on their perimeter distance
        sorted_points = sorted(points, key=lambda p: perimeter_distance(p))
        
        # Binary search between low and high
        low = 0
        high = 2 * side
        answer = 0
        
        while low <= high:
            mid = (low + high) // 2
            selected = []
            possible = False
            for p in sorted_points:
                valid = True
                for s_p in selected:
                    if abs(p[0] - s_p[0]) + abs(p[1] - s_p[1]) < mid:
                        valid = False
                        break
                if valid:
                    selected.append(p)
                    if len(selected) == k:
                        possible = True
                        break
            if possible:
                answer = mid
                low = mid + 1
            else:
                high = mid - 1
        
        return answer