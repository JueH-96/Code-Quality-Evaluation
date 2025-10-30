class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        # Check if we can make two horizontal cuts
        def canMakeHorizontalCuts():
            # Get all unique y coordinates
            y_coords = set()
            for rect in rectangles:
                y_coords.add(rect[1])  # start_y
                y_coords.add(rect[3])  # end_y
            
            y_coords = sorted(list(y_coords))
            
            # Try all pairs of cuts
            for i in range(len(y_coords)):
                for j in range(i + 1, len(y_coords)):
                    cut1 = y_coords[i]
                    cut2 = y_coords[j]
                    
                    # Check if these cuts create 3 non-empty sections
                    section1 = []  # y < cut1
                    section2 = []  # cut1 <= y < cut2
                    section3 = []  # y >= cut2
                    
                    valid = True
                    for rect in rectangles:
                        start_y, end_y = rect[1], rect[3]
                        
                        if end_y <= cut1:
                            section1.append(rect)
                        elif start_y >= cut2:
                            section3.append(rect)
                        elif start_y >= cut1 and end_y <= cut2:
                            section2.append(rect)
                        else:
                            # Rectangle crosses a cut line
                            valid = False
                            break
                    
                    if valid and len(section1) > 0 and len(section2) > 0 and len(section3) > 0:
                        return True
            
            return False
        
        # Check if we can make two vertical cuts
        def canMakeVerticalCuts():
            # Get all unique x coordinates
            x_coords = set()
            for rect in rectangles:
                x_coords.add(rect[0])  # start_x
                x_coords.add(rect[2])  # end_x
            
            x_coords = sorted(list(x_coords))
            
            # Try all pairs of cuts
            for i in range(len(x_coords)):
                for j in range(i + 1, len(x_coords)):
                    cut1 = x_coords[i]
                    cut2 = x_coords[j]
                    
                    # Check if these cuts create 3 non-empty sections
                    section1 = []  # x < cut1
                    section2 = []  # cut1 <= x < cut2
                    section3 = []  # x >= cut2
                    
                    valid = True
                    for rect in rectangles:
                        start_x, end_x = rect[0], rect[2]
                        
                        if end_x <= cut1:
                            section1.append(rect)
                        elif start_x >= cut2:
                            section3.append(rect)
                        elif start_x >= cut1 and end_x <= cut2:
                            section2.append(rect)
                        else:
                            # Rectangle crosses a cut line
                            valid = False
                            break
                    
                    if valid and len(section1) > 0 and len(section2) > 0 and len(section3) > 0:
                        return True
            
            return False
        
        return canMakeHorizontalCuts() or canMakeVerticalCuts()