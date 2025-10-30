from itertools import combinations

class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        max_area = -1
        n = len(points)
        if n < 4:
            return -1

        def is_rectangle(rect_points):
            x_coords = set()
            y_coords = set()
            for p in rect_points:
                x_coords.add(p[0])
                y_coords.add(p[1])
            if len(x_coords) != 2 or len(y_coords) != 2:
                return False
            x_vals = sorted(list(x_coords))
            y_vals = sorted(list(y_coords))
            expected_points = set([(x_vals[0], y_vals[0]), (x_vals[0], y_vals[1]), (x_vals[1], y_vals[0]), (x_vals[1], y_vals[1])])
            given_points = set(tuple(p) for p in rect_points)
            return expected_points == given_points

        def is_inside_rectangle(rect_points, point):
            x_coords = [p[0] for p in rect_points]
            y_coords = [p[1] for p in rect_points]
            min_x = min(x_coords)
            max_x = max(x_coords)
            min_y = min(y_coords)
            max_y = max(y_coords)
            point_x, point_y = point
            return min_x <= point_x <= max_x and min_y <= point_y <= max_y

        for rect_points_list in combinations(points, 4):
            if is_rectangle(list(rect_points_list)):
                valid_rectangle = True
                rect_x_coords = [p[0] for p in rect_points_list]
                rect_y_coords = [p[1] for p in rect_points_list]
                min_x = min(rect_x_coords)
                max_x = max(rect_x_coords)
                min_y = min(rect_y_coords)
                max_y = max(rect_y_coords)

                for other_point in points:
                    if list(other_point) not in list(rect_points_list):
                        if is_inside_rectangle(list(rect_points_list), other_point):
                            valid_rectangle = False
                            break
                if valid_rectangle:
                    area = (max_x - min_x) * (max_y - min_y)
                    max_area = max(max_area, area)

        return max_area