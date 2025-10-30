class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        point_set = set(map(tuple, points))
        max_area = -1
        
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                x1, y1 = points[i]
                x2, y2 = points[j]
                
                if x1 != x2 and y1 != y2:
                    if (x1, y2) in point_set and (x2, y1) in point_set:
                        area = abs(x2 - x1) * abs(y2 - y1)
                        if self.isRectangleEmpty(points, x1, y1, x2, y2):
                            max_area = max(max_area, area)
        
        return max_area
    
    def isRectangleEmpty(self, points, x1, y1, x2, y2):
        for x, y in points:
            if x1 < x < x2 and y1 < y < y2:
                return False
        return True