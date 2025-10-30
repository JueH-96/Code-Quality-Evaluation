class Solution:
    def maxDistance(self, side: int, points: List[List[int]], k: int) -> int:
        def manhattan_distance(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
        def can_select_k_points(min_dist):
            # Start with first point
            selected = [points[0]]
            
            # Try to select k points with minimum distance >= min_dist
            for point in points[1:]:
                # Check if current point can be selected
                valid = True
                for sel in selected:
                    if manhattan_distance(point, sel) < min_dist:
                        valid = False
                        break
                
                if valid:
                    selected.append(point)
                    if len(selected) >= k:
                        return True
            
            return False
        
        # Binary search for the maximum possible minimum distance
        left, right = 0, 2 * side  # Maximum possible Manhattan distance is 2*side
        
        while left < right:
            mid = (left + right + 1) // 2
            
            if can_select_k_points(mid):
                left = mid
            else:
                right = mid - 1
                
        return left