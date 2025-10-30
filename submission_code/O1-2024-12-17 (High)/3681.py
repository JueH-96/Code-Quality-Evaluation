class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        # Convert points to a set for O(1) membership checks
        point_set = set(tuple(p) for p in points)

        # Collect distinct x and y coordinates
        xs = sorted(set(p[0] for p in points))
        ys = sorted(set(p[1] for p in points))

        max_area = -1
        n = len(points)

        # Helper to check if any extra point lies inside or on boundary
        def has_inner_point(x1, y1, x2, y2):
            for px, py in points:
                # If within the rectangle boundaries (including edges)
                if x1 <= px <= x2 and y1 <= py <= y2:
                    # This point must be one of the four corners to be valid
                    if (px, py) not in {
                        (x1, y1), (x1, y2), (x2, y1), (x2, y2)
                    }:
                        return True
            return False

        # Try all pairs of distinct x coordinates
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                x1, x2 = xs[i], xs[j]
                # Try all pairs of distinct y coordinates
                for p in range(len(ys)):
                    for q in range(p + 1, len(ys)):
                        y1, y2 = ys[p], ys[q]

                        # Check if the 4 corners exist
                        c1 = (x1, y1)
                        c2 = (x1, y2)
                        c3 = (x2, y1)
                        c4 = (x2, y2)
                        if c1 in point_set and c2 in point_set and c3 in point_set and c4 in point_set:
                            # Check if no other point is inside or on boundary
                            if not has_inner_point(x1, y1, x2, y2):
                                area = (x2 - x1) * (y2 - y1)
                                max_area = max(max_area, area)

        return max_area