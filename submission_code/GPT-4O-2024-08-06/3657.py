class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        # Collect all unique x and y coordinates from the rectangles
        x_coords = set()
        y_coords = set()
        
        for rect in rectangles:
            start_x, start_y, end_x, end_y = rect
            x_coords.add(start_x)
            x_coords.add(end_x)
            y_coords.add(start_y)
            y_coords.add(end_y)
        
        # Convert sets to sorted lists
        x_coords = sorted(x_coords)
        y_coords = sorted(y_coords)
        
        # Check for possible vertical cuts
        for i in range(1, len(x_coords) - 1):
            left = x_coords[:i]
            right = x_coords[i:]
            
            left_rects = [rect for rect in rectangles if rect[2] <= left[-1]]
            middle_rects = [rect for rect in rectangles if rect[0] >= left[-1] and rect[2] <= right[0]]
            right_rects = [rect for rect in rectangles if rect[0] >= right[0]]
            
            if left_rects and middle_rects and right_rects:
                return True
        
        # Check for possible horizontal cuts
        for i in range(1, len(y_coords) - 1):
            bottom = y_coords[:i]
            top = y_coords[i:]
            
            bottom_rects = [rect for rect in rectangles if rect[3] <= bottom[-1]]
            middle_rects = [rect for rect in rectangles if rect[1] >= bottom[-1] and rect[3] <= top[0]]
            top_rects = [rect for rect in rectangles if rect[1] >= top[0]]
            
            if bottom_rects and middle_rects and top_rects:
                return True
        
        return False