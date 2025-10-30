class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        # Collect all unique x and y coordinates from the rectangles
        x_coords = set()
        y_coords = set()
        
        for rect in rectangles:
            x_coords.add(rect[0])
            x_coords.add(rect[2])
            y_coords.add(rect[1])
            y_coords.add(rect[3])
        
        x_coords = sorted(x_coords)
        y_coords = sorted(y_coords)
        
        # Check for vertical cuts
        for i in range(1, len(x_coords) - 1):
            for j in range(i + 1, len(x_coords) - 1):
                left = x_coords[:i]
                middle = x_coords[i:j]
                right = x_coords[j:]
                
                left_rects = [rect for rect in rectangles if rect[2] <= left[-1]]
                middle_rects = [rect for rect in rectangles if rect[0] >= middle[0] and rect[2] <= middle[-1]]
                right_rects = [rect for rect in rectangles if rect[0] >= right[0]]
                
                if left_rects and middle_rects and right_rects:
                    return True
        
        # Check for horizontal cuts
        for i in range(1, len(y_coords) - 1):
            for j in range(i + 1, len(y_coords) - 1):
                bottom = y_coords[:i]
                middle = y_coords[i:j]
                top = y_coords[j:]
                
                bottom_rects = [rect for rect in rectangles if rect[3] <= bottom[-1]]
                middle_rects = [rect for rect in rectangles if rect[1] >= middle[0] and rect[3] <= middle[-1]]
                top_rects = [rect for rect in rectangles if rect[1] >= top[0]]
                
                if bottom_rects and middle_rects and top_rects:
                    return True
        
        return False