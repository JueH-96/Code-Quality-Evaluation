class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        def check_cuts(dimension):
            # Collect all unique coordinates for the given dimension
            bounds = set()
            for rect in rectangles:
                bounds.add(rect[dimension])        # Start coordinate
                bounds.add(rect[dimension + 2])    # End coordinate
            
            bounds = sorted(list(bounds))
            
            # Try all possible pairs of cuts
            for i in range(len(bounds) - 1):
                for j in range(i + 1, len(bounds)):
                    cut1, cut2 = bounds[i], bounds[j]
                    
                    # Count rectangles in each section
                    below, between, above = 0, 0, 0
                    valid = True
                    
                    for rect in rectangles:
                        if rect[dimension + 2] <= cut1:  # Rectangle is below/left of cut1
                            below += 1
                        elif rect[dimension] >= cut2:    # Rectangle is above/right of cut2
                            above += 1
                        elif rect[dimension] >= cut1 and rect[dimension + 2] <= cut2:  # Between cuts
                            between += 1
                        else:  # Rectangle intersects a cut
                            valid = False
                            break
                    
                    if valid and below > 0 and between > 0 and above > 0:
                        return True
            
            return False
        
        # Check for horizontal cuts (dimension 1 is y)
        if check_cuts(1):
            return True
        
        # Check for vertical cuts (dimension 0 is x)
        if check_cuts(0):
            return True
        
        return False