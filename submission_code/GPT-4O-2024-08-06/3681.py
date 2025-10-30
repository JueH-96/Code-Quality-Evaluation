class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        # Convert the list of points to a set for O(1) lookups
        point_set = set(map(tuple, points))
        max_area = -1
        
        # Iterate over all pairs of points to find potential diagonals of rectangles
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                x1, y1 = points[i]
                x2, y2 = points[j]
                
                # Check if these two points can be diagonally opposite corners of a rectangle
                if x1 != x2 and y1 != y2:
                    # Check if the other two corners of the rectangle exist
                    if (x1, y2) in point_set and (x2, y1) in point_set:
                        # Calculate the area of the rectangle
                        area = abs(x2 - x1) * abs(y2 - y1)
                        # Update max_area if this is the largest found so far
                        max_area = max(max_area, area)
        
        return max_area