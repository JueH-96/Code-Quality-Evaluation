class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        points_set = set(tuple(p) for p in points)
        max_area = 0
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                x1, y1 = points[i]
                x2, y2 = points[j]
                if x1 == x2 or y1 == y2:
                    continue
                x3, y3 = x1, y2
                x4, y4 = x2, y1
                if (x3, y3) in points_set and (x4, y4) in points_set:
                    area = abs(x1 - x2) * abs(y1 - y2)
                    valid = True
                    for x in range(min(x1, x2), max(x1, x2) + 1):
                        for y in range(min(y1, y2), max(y1, y2) + 1):
                            if (x, y) in points_set and (x,y) != (x1,y1) and (x,y) != (x2,y2) and (x,y) != (x3,y3) and (x,y) != (x4,y4):
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        max_area = max(max_area, area)
        return max_area