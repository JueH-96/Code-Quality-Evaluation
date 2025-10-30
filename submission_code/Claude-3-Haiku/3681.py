class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        if len(points) < 4:
            return -1
        
        # Sort the points by x-coordinate, then y-coordinate
        points.sort()
        
        max_area = 0
        
        # Iterate through all possible rectangles
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                for k in range(j+1, len(points)):
                    for l in range(k+1, len(points)):
                        # Check if the four points form a valid rectangle
                        if (points[i][0] == points[j][0] and points[j][1] == points[l][1] and
                            points[j][0] == points[k][0] and points[i][1] == points[k][1]):
                            # Calculate the area of the rectangle
                            area = abs(points[j][0] - points[i][0]) * abs(points[l][1] - points[i][1])
                            max_area = max(max_area, area)
        
        return max_area if max_area > 0 else -1