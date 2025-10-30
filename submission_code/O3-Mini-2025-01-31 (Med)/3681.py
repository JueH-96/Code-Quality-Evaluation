from typing import List

class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        # Convert list of points to a set for O(1) membership checking.
        pt_set = {(x, y) for x, y in points}
        
        # Extract unique x and y coordinates
        xs = sorted({x for x, _ in points})
        ys = sorted({y for _, y in points})
        
        max_area = -1
        
        # Iterate over all possible pairs of distinct x and y coordinates.
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                x1, x2 = xs[i], xs[j]
                for k in range(len(ys)):
                    for l in range(k + 1, len(ys)):
                        y1, y2 = ys[k], ys[l]
                        
                        # Check if the four corners exist in the set.
                        if ((x1, y1) in pt_set and (x1, y2) in pt_set and 
                            (x2, y1) in pt_set and (x2, y2) in pt_set):
                            # Now check that no other point lies on or inside the rectangle.
                            valid = True
                            for px, py in pt_set:
                                # If the point is one of the corners, skip.
                                if (px, py) in {(x1, y1), (x1, y2), (x2, y1), (x2, y2)}:
                                    continue
                                # If a point lies within the boundary (inclusive) of the rectangle, invalidate.
                                if x1 <= px <= x2 and y1 <= py <= y2:
                                    valid = False
                                    break
                            if valid:
                                area = (x2 - x1) * (y2 - y1)
                                if area > max_area:
                                    max_area = area
        return max_area