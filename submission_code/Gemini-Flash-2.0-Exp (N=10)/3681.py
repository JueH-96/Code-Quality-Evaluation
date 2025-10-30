class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        n = len(points)
        if n < 4:
            return -1

        points_set = set(tuple(p) for p in points)
        max_area = -1

        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]

                if x1 == x2 or y1 == y2:
                    continue

                x3, y3 = x1, y2
                x4, y4 = x2, y1

                if (x3, y3) in points_set and (x4, y4) in points_set:
                    
                    is_valid = True
                    min_x = min(x1,x2)
                    max_x = max(x1,x2)
                    min_y = min(y1,y2)
                    max_y = max(y1,y2)

                    for k in range(n):
                        if k == i or k == j:
                            continue
                        xk, yk = points[k]
                        if min_x < xk < max_x and min_y < yk < max_y:
                            is_valid = False
                            break
                        if (xk == min_x or xk == max_x) and (min_y <= yk <= max_y):
                            if (xk,yk) != (x1,y1) and (xk,yk) != (x2,y2) and (xk,yk) != (x3,y3) and (xk,yk) != (x4,y4):
                                is_valid = False
                                break
                        if (yk == min_y or yk == max_y) and (min_x <= xk <= max_x):
                            if (xk,yk) != (x1,y1) and (xk,yk) != (x2,y2) and (xk,yk) != (x3,y3) and (xk,yk) != (x4,y4):
                                is_valid = False
                                break
                    
                    if is_valid:
                        area = abs(x1 - x2) * abs(y1 - y2)
                        max_area = max(max_area, area)

        return max_area