from typing import List

class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        # Convert list of points to a set for O(1) lookups
        pts = set((x, y) for x, y in points)
        n = len(points)
        max_area = 0

        # Try every pair of points as potential diagonal corners
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                # They must differ in both coordinates to form a diagonal
                if x1 == x2 or y1 == y2:
                    continue
                # Check the other two corners of the axis-aligned rectangle
                if (x1, y2) in pts and (x2, y1) in pts:
                    minX, maxX = min(x1, x2), max(x1, x2)
                    minY, maxY = min(y1, y2), max(y1, y2)
                    
                    # Ensure no other point lies inside or on the border
                    valid = True
                    for x, y in points:
                        # Skip the four corners themselves
                        if (x, y) in ((x1, y1), (x1, y2), (x2, y1), (x2, y2)):
                            continue
                        # If any other point is within the bounding box, invalidate
                        if minX <= x <= maxX and minY <= y <= maxY:
                            valid = False
                            break
                    
                    if valid:
                        area = (maxX - minX) * (maxY - minY)
                        max_area = max(max_area, area)

        # If no valid rectangle was found, return -1
        return max_area if max_area > 0 else -1