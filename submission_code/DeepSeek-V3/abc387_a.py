class Solution:
    def maxDistance(self, side: int, points: List[List[int]], k: int) -> int:
        # Step 1: Sort the points in the order of traversing the perimeter clockwise
        perimeter_points = []
        
        # Top edge: (x, side) where x ranges from 0 to side (left to right)
        top = [p for p in points if p[1] == side and p[0] != side]
        top.sort()
        perimeter_points.extend(top)
        
        # Right edge: (side, y) where y ranges from side to 0 (top to bottom), excluding (side, side) if it was added in top
        right = [p for p in points if p[0] == side and p[1] != 0]
        right.sort(key=lambda x: -x[1])
        perimeter_points.extend(right)
        
        # Bottom edge: (x, 0) where x ranges from side to 0 (right to left), excluding (side, 0)
        bottom = [p for p in points if p[1] == 0 and p[0] != 0]
        bottom.sort(key=lambda x: -x[0])
        perimeter_points.extend(bottom)
        
        # Left edge: (0, y) where y ranges from 0 to side (bottom to top), excluding (0, 0)
        left = [p for p in points if p[0] == 0 and p[1] != side]
        left.sort(key=lambda x: x[1])
        perimeter_points.extend(left)
        
        # Handle the corners: (0,0), (0, side), (side, side), (side, 0)
        # Check if any corner points are in the list and place them in the correct order
        corners_order = [
            (0, side), (side, side), (side, 0), (0, 0)
        ]
        # We need to insert the corners in the correct positions if they exist in the points list
        # But since the perimeter_points already includes all points except possibly some corners, we need to re-sort them properly
        # Alternative approach: collect all points and sort them according to their position along the perimeter
        
        # Re-defining the perimeter_points properly
        # The perimeter order is: start at (0,0), move right to (side,0), up to (side,side), left to (0,side), down to (0,0)
        # So for each point, compute its position along the perimeter
        def get_perimeter_position(p):
            x, y = p
            if y == 0:
                return x  # 0 to side
            elif x == side:
                return side + y  # side to side + side
            elif y == side:
                return 3 * side - x  # side + side to 3*side - side = 2*side (when x=0)
            else:  # x == 0
                return 4 * side - y  # 3*side to 4*side - side=3*side (when y=0)
        
        perimeter_points = sorted(points, key=lambda p: get_perimeter_position(p))
        
        # Now proceed with binary search
        low = 0
        high = 2 * side
        best = 0
        
        def is_possible(d):
            count = 1
            last = perimeter_points[0]
            for i in range(1, len(perimeter_points)):
                current = perimeter_points[i]
                # Compute Manhattan distance between last and current
                distance = abs(current[0] - last[0]) + abs(current[1] - last[1])
                if distance >= d:
                    count += 1
                    last = current
                    if count >= k:
                        return True
            return count >= k
        
        while low <= high:
            mid = (low + high) // 2
            if is_possible(mid):
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        return best