class Solution:
    def maxDistance(self, side: int, points: List[List[int]], k: int) -> int:
        # Convert boundary points to 1D positions along the perimeter
        def get_perimeter_pos(x, y):
            if x == 0:  # Left edge
                return y
            elif y == side:  # Top edge
                return side + x
            elif x == side:  # Right edge
                return 2 * side + (side - y)
            else:  # Bottom edge (y == 0)
                return 3 * side + (side - x)
        
        # Convert all points to perimeter positions
        perimeter_points = []
        for x, y in points:
            pos = get_perimeter_pos(x, y)
            perimeter_points.append((pos, x, y))
        
        # Sort by perimeter position
        perimeter_points.sort()
        
        # Function to calculate Manhattan distance
        def manhattan_dist(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
        # Check if we can select k points with minimum distance >= min_dist
        def can_select(min_dist):
            selected = [perimeter_points[0]]  # Always select first point
            
            for i in range(1, len(perimeter_points)):
                # Check if current point is far enough from all selected points
                valid = True
                for selected_point in selected:
                    if manhattan_dist(perimeter_points[i][1:], selected_point[1:]) < min_dist:
                        valid = False
                        break
                
                if valid:
                    selected.append(perimeter_points[i])
                    if len(selected) == k:
                        return True
            
            return len(selected) >= k
        
        # Binary search on the answer
        left, right = 0, 4 * side
        result = 0
        
        while left <= right:
            mid = (left + right) // 2
            if can_select(mid):
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return result