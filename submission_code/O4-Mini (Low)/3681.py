from typing import List

class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        # Convert list of points to a set for O(1) lookup
        pts_set = set((x, y) for x, y in points)
        n = len(points)
        max_area = -1

        # Iterate over all pairs of distinct x-coordinates
        xs = sorted({x for x, y in points})
        ys = sorted({y for x, y in points})
        # Try every combination of x1 < x2 and y1 < y2
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                x1, x2 = xs[i], xs[j]
                for a in range(len(ys)):
                    for b in range(a + 1, len(ys)):
                        y1, y2 = ys[a], ys[b]
                        # Check that the four corners exist
                        if (x1, y1) in pts_set and (x1, y2) in pts_set and \
                           (x2, y1) in pts_set and (x2, y2) in pts_set:
                            # Check no other point lies inside or on border
                            valid = True
                            for px, py in points:
                                if (px, py) in {(x1, y1), (x1, y2), (x2, y1), (x2, y2)}:
                                    continue
                                # If point is within or on the edges of the rectangle
                                if x1 <= px <= x2 and y1 <= py <= y2:
                                    valid = False
                                    break
                            if valid:
                                area = (x2 - x1) * (y2 - y1)
                                if area > max_area:
                                    max_area = area

        return max_area