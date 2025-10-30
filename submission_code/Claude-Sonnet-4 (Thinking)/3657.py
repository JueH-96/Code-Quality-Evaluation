class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        # Try horizontal cuts
        y_coords = set()
        for rect in rectangles:
            y_coords.add(rect[1])  # start_y
            y_coords.add(rect[3])  # end_y
        
        y_coords = sorted(y_coords)
        
        for i in range(len(y_coords)):
            for j in range(i + 1, len(y_coords)):
                y1, y2 = y_coords[i], y_coords[j]
                
                section1, section2, section3 = 0, 0, 0
                valid = True
                
                for rect in rectangles:
                    start_y, end_y = rect[1], rect[3]
                    
                    count = 0
                    if end_y <= y1:
                        count += 1
                        section1 += 1
                    if start_y >= y1 and end_y <= y2:
                        count += 1
                        section2 += 1
                    if start_y >= y2:
                        count += 1
                        section3 += 1
                    
                    if count != 1:  # Rectangle doesn't belong to exactly one section
                        valid = False
                        break
                
                if valid and section1 > 0 and section2 > 0 and section3 > 0:
                    return True
        
        # Try vertical cuts
        x_coords = set()
        for rect in rectangles:
            x_coords.add(rect[0])  # start_x
            x_coords.add(rect[2])  # end_x
        
        x_coords = sorted(x_coords)
        
        for i in range(len(x_coords)):
            for j in range(i + 1, len(x_coords)):
                x1, x2 = x_coords[i], x_coords[j]
                
                section1, section2, section3 = 0, 0, 0
                valid = True
                
                for rect in rectangles:
                    start_x, end_x = rect[0], rect[2]
                    
                    count = 0
                    if end_x <= x1:
                        count += 1
                        section1 += 1
                    if start_x >= x1 and end_x <= x2:
                        count += 1
                        section2 += 1
                    if start_x >= x2:
                        count += 1
                        section3 += 1
                    
                    if count != 1:  # Rectangle doesn't belong to exactly one section
                        valid = False
                        break
                
                if valid and section1 > 0 and section2 > 0 and section3 > 0:
                    return True
        
        return False