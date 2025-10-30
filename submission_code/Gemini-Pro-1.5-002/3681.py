class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        n = len(points)
        if n < 4:
            return -1
        
        point_set = set()
        for x, y in points:
            point_set.add((x, y))
            
        max_area = -1
        
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                
                if x1 != x2 and y1 != y2:
                    if (x1, y2) in point_set and (x2, y1) in point_set:
                        area = abs(x1 - x2) * abs(y1 - y2)
                        
                        valid = True
                        for x in range(min(x1, x2) + 1, max(x1, x2)):
                            for y in range(min(y1, y2) + 1, max(y1, y2)):
                                if (x, y) in point_set:
                                    valid = False
                                    break
                            if not valid:
                                break
                        
                        if valid:
                            if max_area == -1 or area < max_area:
                                max_area = area
                                
        return max_area