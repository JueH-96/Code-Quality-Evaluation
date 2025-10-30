class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        # Convert list of points to a set for O(1) membership checks
        point_set = set(tuple(p) for p in points)

        n = len(points)
        max_area = -1

        # Try all distinct pairs of points
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]

                # Determine the min/max of x and y to form potential rectangle corners
                left, right = min(x1, x2), max(x1, x2)
                bottom, top = min(y1, y2), max(y1, y2)

                # If this doesn't form a rectangle with non-zero area, skip
                if left == right or bottom == top:
                    continue

                # Identify the four corners
                corners = {
                    (left, bottom),
                    (left, top),
                    (right, bottom),
                    (right, top)
                }

                # Check if all corners are in our point set
                if not corners.issubset(point_set):
                    continue

                # Check that no extra points lie on or inside this rectangle
                valid = True
                for px, py in points:
                    # If this point is within or on the boundary of the rectangle
                    if left <= px <= right and bottom <= py <= top:
                        # It must be one of the four corners; otherwise, it's invalid
                        if (px, py) not in corners:
                            valid = False
                            break

                if valid:
                    # Compute area
                    area = (right - left) * (top - bottom)
                    max_area = max(max_area, area)

        return max_area