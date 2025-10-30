class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        point_set = set()
        for p in points:
            point_set.add(tuple(p))

        max_area = -1
        n = len(points)
        if n < 4:
            return -1

        for i in range(n):
            for j in range(i + 1, n):
                p1 = points[i]
                p2 = points[j]
                x1, y1 = p1[0], p1[1]
                x2, y2 = p2[0], p2[1]

                if x1 == x2 or y1 == y2:
                    continue

                p3 = (x1, y2)
                p4 = (x2, y1)

                if p3 in point_set and p4 in point_set:
                    current_points = {tuple(p1), tuple(p2), p3, p4}
                    valid_rectangle = True
                    min_x = min(x1, x2)
                    max_x = max(x1, x2)
                    min_y = min(y1, y2)
                    max_y = max(y1, y2)
                    
                    for point in points:
                        pt = tuple(point)
                        if pt not in current_points and min_x <= point[0] <= max_x and min_y <= point[1] <= max_y:
                            valid_rectangle = False
                            break
                    
                    if valid_rectangle:
                        area = abs(x1 - x2) * abs(y1 - y2)
                        max_area = max(max_area, area)

        return max_area