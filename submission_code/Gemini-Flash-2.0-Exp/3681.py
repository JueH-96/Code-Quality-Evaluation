class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        n = len(points)
        max_area = -1
        
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        
                        x_coords = [points[i][0], points[j][0], points[k][0], points[l][0]]
                        y_coords = [points[i][1], points[j][1], points[k][1], points[l][1]]
                        
                        x_coords.sort()
                        y_coords.sort()
                        
                        if x_coords[0] == x_coords[1] and x_coords[2] == x_coords[3] and y_coords[0] == y_coords[1] and y_coords[2] == y_coords[3] and x_coords[0] != x_coords[2] and y_coords[0] != y_coords[2]:
                            
                            x1 = x_coords[0]
                            x2 = x_coords[2]
                            y1 = y_coords[0]
                            y2 = y_coords[2]
                            
                            area = abs(x2 - x1) * abs(y2 - y1)
                            
                            valid = True
                            for m in range(n):
                                if m not in [i, j, k, l]:
                                    x = points[m][0]
                                    y = points[m][1]
                                    
                                    if x > min(x1, x2) and x < max(x1, x2) and y > min(y1, y2) and y < max(y1, y2):
                                        valid = False
                                        break
                                    elif (x == min(x1,x2) or x == max(x1,x2)) and (y >= min(y1,y2) and y <= max(y1,y2)):
                                        valid = False
                                        break
                                    elif (y == min(y1,y2) or y == max(y1,y2)) and (x >= min(x1,x2) and x <= max(x1,x2)):
                                        valid = False
                                        break
                            
                            if valid:
                                max_area = max(max_area, area)
        
        return max_area