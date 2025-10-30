class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        n = len(points)
        max_area = -1
        points_set = set(tuple(p) for p in points)

        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        x1, y1 = points[i]
                        x2, y2 = points[j]
                        x3, y3 = points[k]
                        x4, y4 = points[l]

                        xs = sorted([x1, x2, x3, x4])
                        ys = sorted([y1, y2, y3, y4])

                        if xs[0] == xs[1] and xs[2] == xs[3] and ys[0] == ys[1] and ys[2] == ys[3] and xs[0] != xs[2] and ys[0] != ys[2]:
                            
                            if (xs[0], ys[0]) in points_set and (xs[0], ys[2]) in points_set and (xs[2], ys[0]) in points_set and (xs[2], ys[2]) in points_set:
                                valid = True
                                for m in range(n):
                                    if m not in (i, j, k, l):
                                        xm, ym = points[m]
                                        if xs[0] <= xm <= xs[2] and ys[0] <= ym <= ys[2]:
                                            if not (xm == xs[0] or xm == xs[2]) or not (ym == ys[0] or ym == ys[2]):
                                                valid = False
                                                break
                                if valid:
                                    area = (xs[2] - xs[0]) * (ys[2] - ys[0])
                                    max_area = max(max_area, area)

        return max_area