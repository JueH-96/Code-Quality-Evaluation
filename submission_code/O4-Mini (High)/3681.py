from typing import List

class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        # Convert to a set of tuples for O(1) lookups
        pts = set((x, y) for x, y in points)
        # Extract sorted unique x-coordinates and y-coordinates
        xs = sorted({x for x, y in points})
        ys = sorted({y for x, y in points})
        
        max_area = -1
        
        # Try every pair of distinct x's and every pair of distinct y's
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                x1, x2 = xs[i], xs[j]
                for k in range(len(ys)):
                    for l in range(k + 1, len(ys)):
                        y1, y2 = ys[k], ys[l]
                        
                        # Define the four corners of the candidate rectangle
                        corners = {(x1, y1), (x1, y2), (x2, y1), (x2, y2)}
                        
                        # Check if all four corners are present
                        if not corners.issubset(pts):
                            continue
                        
                        # Verify that no other point lies inside or on the border
                        valid = True
                        for px, py in pts:
                            if (px, py) in corners:
                                continue
                            if x1 <= px <= x2 and y1 <= py <= y2:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        # Compute area and update maximum
                        area = (x2 - x1) * (y2 - y1)
                        if area > max_area:
                            max_area = area
        
        return max_area