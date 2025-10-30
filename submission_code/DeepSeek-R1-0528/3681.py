class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        n = len(points)
        point_set = set(tuple(p) for p in points)
        max_area = -1
        
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                if x1 < x2 and y1 < y2:
                    if (x1, y2) in point_set and (x2, y1) in point_set:
                        corners = {(x1, y1), (x1, y2), (x2, y1), (x2, y2)}
                        valid = True
                        for p in points:
                            x, y = p
                            if x1 <= x <= x2 and y1 <= y <= y2:
                                if (x, y) not in corners:
                                    valid = False
                                    break
                        if valid:
                            area = (x2 - x1) * (y2 - y1)
                            if area > max_area:
                                max_area = area
        return max_area