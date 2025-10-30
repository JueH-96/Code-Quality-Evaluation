class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        from itertools import combinations

        max_area = -1
        point_set = set(map(tuple, points))

        n = len(points)
        for quad in combinations(points, 4):
            x_coords = set([p[0] for p in quad])
            y_coords = set([p[1] for p in quad])

            if len(x_coords) != 2 or len(y_coords) != 2:
                continue  # Not a rectangle with sides parallel to axes

            x1, x2 = sorted(x_coords)
            y1, y2 = sorted(y_coords)

            # Check that all four corner points are present
            required_corners = {(x1, y1), (x1, y2), (x2, y1), (x2, y2)}
            if not required_corners.issubset(set(map(tuple, quad))):
                continue  # Not a proper rectangle

            # Check if any other point lies inside or on the border
            other_points = point_set - set(map(tuple, quad))
            has_point_inside = False
            for px, py in other_points:
                if x1 <= px <= x2 and y1 <= py <= y2:
                    has_point_inside = True
                    break

            if has_point_inside:
                continue  # Rectangle contains other points

            area = (x2 - x1) * (y2 - y1)
            if area > 0:
                max_area = max(max_area, area)

        return max_area