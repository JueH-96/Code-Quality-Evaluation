from itertools import combinations

class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        max_area = -1
        for quad in combinations(points, 4):
            quad_set = set(tuple(p) for p in quad)
            x_coords = [p[0] for p in quad]
            y_coords = [p[1] for p in quad]
            unique_x = set(x_coords)
            unique_y = set(y_coords)
            if len(unique_x) != 2 or len(unique_y) != 2:
                continue
            x1, x2 = sorted(unique_x)
            y1, y2 = sorted(unique_y)
            invalid = False
            for p in points:
                if tuple(p) in quad_set:
                    continue
                x, y = p
                if x1 <= x <= x2 and y1 <= y <= y2:
                    invalid = True
                    break
            if not invalid:
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
        return max_area if max_area != 0 else -1