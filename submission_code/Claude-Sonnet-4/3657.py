class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        def canMakeHorizontalCuts():
            # Sort rectangles by bottom y-coordinate
            rects = sorted(rectangles, key=lambda r: r[1])
            
            # Get all possible cut positions (y-coordinates where rectangles start/end)
            y_coords = set()
            for rect in rectangles:
                y_coords.add(rect[1])  # bottom
                y_coords.add(rect[3])  # top
            
            y_coords = sorted(y_coords)
            
            # Try all pairs of cut positions
            for i in range(len(y_coords)):
                for j in range(i + 1, len(y_coords)):
                    cut1, cut2 = y_coords[i], y_coords[j]
                    
                    # Check if these cuts create valid sections
                    bottom_count = 0
                    middle_count = 0
                    top_count = 0
                    
                    valid = True
                    for rect in rectangles:
                        start_y, end_y = rect[1], rect[3]
                        
                        # Check which section this rectangle belongs to
                        if end_y <= cut1:
                            bottom_count += 1
                        elif start_y >= cut2:
                            top_count += 1
                        elif start_y >= cut1 and end_y <= cut2:
                            middle_count += 1
                        else:
                            # Rectangle spans across cuts - invalid
                            valid = False
                            break
                    
                    if valid and bottom_count > 0 and middle_count > 0 and top_count > 0:
                        return True
            
            return False
        
        def canMakeVerticalCuts():
            # Sort rectangles by left x-coordinate
            rects = sorted(rectangles, key=lambda r: r[0])
            
            # Get all possible cut positions (x-coordinates where rectangles start/end)
            x_coords = set()
            for rect in rectangles:
                x_coords.add(rect[0])  # left
                x_coords.add(rect[2])  # right
            
            x_coords = sorted(x_coords)
            
            # Try all pairs of cut positions
            for i in range(len(x_coords)):
                for j in range(i + 1, len(x_coords)):
                    cut1, cut2 = x_coords[i], x_coords[j]
                    
                    # Check if these cuts create valid sections
                    left_count = 0
                    middle_count = 0
                    right_count = 0
                    
                    valid = True
                    for rect in rectangles:
                        start_x, end_x = rect[0], rect[2]
                        
                        # Check which section this rectangle belongs to
                        if end_x <= cut1:
                            left_count += 1
                        elif start_x >= cut2:
                            right_count += 1
                        elif start_x >= cut1 and end_x <= cut2:
                            middle_count += 1
                        else:
                            # Rectangle spans across cuts - invalid
                            valid = False
                            break
                    
                    if valid and left_count > 0 and middle_count > 0 and right_count > 0:
                        return True
            
            return False
        
        return canMakeHorizontalCuts() or canMakeVerticalCuts()