from typing import List
from collections import defaultdict

class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        point_set = set((x, y) for x, y in points)
        x_to_ys = defaultdict(set)
        for x, y in points:
            x_to_ys[x].add(y)
        xs = sorted(x_to_ys.keys())
        max_area = -1
        
        for i in range(len(xs)):
            x1 = xs[i]
            for j in range(i + 1, len(xs)):
                x2 = xs[j]
                ys1 = x_to_ys[x1]
                ys2 = x_to_ys[x2]
                common_ys = sorted(ys1.intersection(ys2))
                if len(common_ys) < 2:
                    continue
                for k in range(len(common_ys)):
                    for l in range(k + 1, len(common_ys)):
                        y1 = common_ys[k]
                        y2 = common_ys[l]
                        is_valid = True
                        for (x, y) in points:
                            if x >= x1 and x <= x2 and y >= y1 and y <= y2:
                                if (x == x1 or x == x2) and (y == y1 or y == y2):
                                    continue
                                else:
                                    is_valid = False
                                    break
                        if is_valid:
                            area = (x2 - x1) * (y2 - y1)
                            if area > max_area:
                                max_area = area
        return max_area if max_area != -1 else -1