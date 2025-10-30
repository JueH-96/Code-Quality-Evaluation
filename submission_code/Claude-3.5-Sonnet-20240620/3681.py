class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        point_set = set(map(tuple, points))
        n = len(points)
        max_area = -1

        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]

                # Skip if points are on the same vertical or horizontal line
                if x1 == x2 or y1 == y2:
                    continue

                # Check if the other two corners exist
                if (x1, y2) in point_set and (x2, y1) in point_set:
                    # Calculate area
                    area = abs(x1 - x2) * abs(y1 - y2)

                    # Check if there are any points inside the rectangle
                    is_valid = True
                    for x, y in points:
                        if (x1 < x < x2 or x2 < x < x1) and (y1 < y < y2 or y2 < y < y1):
                            is_valid = False
                            break

                    if is_valid:
                        max_area = max(max_area, area)

        return max_area