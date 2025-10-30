class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        # Try horizontal cuts
        def try_horizontal_cuts():
            # Get all unique y-coordinates
            y_coords = set()
            for rect in rectangles:
                y_coords.add(rect[1])
                y_coords.add(rect[3])
            y_coords = sorted(list(y_coords))
            
            # Try all possible pairs of cuts
            for i in range(len(y_coords)-1):
                for j in range(i+1, len(y_coords)-1):
                    cut1, cut2 = y_coords[i], y_coords[j]
                    section1, section2, section3 = 0, 0, 0
                    
                    # Check which section each rectangle belongs to
                    for rect in rectangles:
                        if rect[3] <= cut1:  # Below first cut
                            section1 += 1
                        elif rect[1] >= cut2:  # Above second cut
                            section3 += 1
                        elif rect[1] >= cut1 and rect[3] <= cut2:  # Between cuts
                            section2 += 1
                            
                    # If each section has at least one rectangle
                    if section1 > 0 and section2 > 0 and section3 > 0:
                        return True
            return False
        
        # Try vertical cuts
        def try_vertical_cuts():
            # Get all unique x-coordinates
            x_coords = set()
            for rect in rectangles:
                x_coords.add(rect[0])
                x_coords.add(rect[2])
            x_coords = sorted(list(x_coords))
            
            # Try all possible pairs of cuts
            for i in range(len(x_coords)-1):
                for j in range(i+1, len(x_coords)-1):
                    cut1, cut2 = x_coords[i], x_coords[j]
                    section1, section2, section3 = 0, 0, 0
                    
                    # Check which section each rectangle belongs to
                    for rect in rectangles:
                        if rect[2] <= cut1:  # Left of first cut
                            section1 += 1
                        elif rect[0] >= cut2:  # Right of second cut
                            section3 += 1
                        elif rect[0] >= cut1 and rect[2] <= cut2:  # Between cuts
                            section2 += 1
                            
                    # If each section has at least one rectangle
                    if section1 > 0 and section2 > 0 and section3 > 0:
                        return True
            return False
        
        # Return true if either horizontal or vertical cuts work
        return try_horizontal_cuts() or try_vertical_cuts()