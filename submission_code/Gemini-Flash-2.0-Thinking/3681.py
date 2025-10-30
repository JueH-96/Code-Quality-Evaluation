class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        points_set = set(tuple(p) for p in points)
        n = len(points)
        max_area = -1

        for i in range(n):
            for j in range(i + 1, n):
                p1 = points[i]
                p2 = points[j]
                x1, y1 = p1
                x2, y2 = p2

                if x1 != x2 and y1 != y2:
                    p3 = (x1, y2)
                    p4 = (x2, y1)

                    if tuple(p3) in points_set and tuple(p4) in points_set:
                        area = abs(x1 - x2) * abs(y1 - y2)
                        min_x = min(x1, x2)
                        max_x = max(x1, x2)
                        min_y = min(y1, y2)
                        max_y = max(y1, y2)

                        is_empty = True
                        for px, py in points:
                            if min_x <= px <= max_x and min_y <= py <= max_y:
                                if (px, py) != tuple(p1) and (px, py) != tuple(p2) and (px, py) != tuple(p3) and (px, py) != tuple(p4):
                                    is_empty = False
                                    break

                        if is_empty:
                            max_area = max(max_area, area)

        return max_area