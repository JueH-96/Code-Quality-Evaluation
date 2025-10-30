class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        # Create a set for O(1) lookup of points (as tuples)
        pset = set((x, y) for x, y in points)
        max_area = -1
        n = len(points)
        
        # Since the number of points is small (<=10), we can also get all distinct x and y coordinates.
        xs = sorted(set(x for x, _ in points))
        ys = sorted(set(y for _, y in points))
        
        # Iterate over every pair of distinct x and y to form candidate rectangle corners
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                x1, x2 = xs[i], xs[j]
                for k in range(len(ys)):
                    for l in range(k + 1, len(ys)):
                        y1, y2 = ys[k], ys[l]
                        # Construct candidate rectangle corners
                        corners = {(x1, y1), (x1, y2), (x2, y1), (x2, y2)}
                        # Check if all 4 corners exist in the points set
                        if not corners.issubset(pset):
                            continue
                        
                        # Now, check that there is no other point (not the corners) inside or on the border
                        valid = True
                        for (x, y) in pset:
                            if (x, y) in corners:
                                continue
                            # If the point is lying inside or on the border, it's not valid.
                            # We check if x is between x1 and x2 and y is between y1 and y2.
                            if x1 <= x <= x2 and y1 <= y <= y2:
                                valid = False
                                break
                        if valid:
                            area = (x2 - x1) * (y2 - y1)
                            max_area = max(max_area, area)
        return max_area