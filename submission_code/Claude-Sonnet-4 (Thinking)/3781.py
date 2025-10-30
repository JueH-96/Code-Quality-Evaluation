class Solution:
    def maxDistance(self, side: int, points: List[List[int]], k: int) -> int:
        def parameterize(x, y):
            # Map boundary points to their position going clockwise around square
            if y == 0:  # bottom edge
                return x
            elif x == side:  # right edge
                return side + y
            elif y == side:  # top edge
                return 2 * side + (side - x)
            else:  # left edge (x == 0)
                return 3 * side + (side - y)
        
        def manhattan_distance(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
        def can_select_k_points(d):
            # Sort points by their position on the boundary
            sorted_points = sorted(points, key=lambda p: parameterize(p[0], p[1]))
            
            selected = []
            for point in sorted_points:
                # Check if this point is at distance at least d from all selected points
                if all(manhattan_distance(point, selected_point) >= d for selected_point in selected):
                    selected.append(point)
                    if len(selected) == k:
                        return True
            return False
        
        # Binary search on the answer
        left, right = 0, 2 * side
        while left < right:
            mid = (left + right + 1) // 2
            if can_select_k_points(mid):
                left = mid
            else:
                right = mid - 1
        
        return left