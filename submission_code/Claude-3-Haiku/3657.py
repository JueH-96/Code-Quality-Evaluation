class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        # Sort the rectangles by their start_x and start_y coordinates
        rectangles.sort(key=lambda x: (x[0], x[1]))
        
        # Create sets to store the unique x and y coordinates
        x_coords = set()
        y_coords = set()
        
        # Iterate through the rectangles and add their coordinates to the sets
        for start_x, start_y, end_x, end_y in rectangles:
            x_coords.add(start_x)
            x_coords.add(end_x)
            y_coords.add(start_y)
            y_coords.add(end_y)
        
        # Check if there are at least 3 unique x and y coordinates
        if len(x_coords) < 3 or len(y_coords) < 3:
            return False
        
        # Sort the x and y coordinates
        x_coords = sorted(x_coords)
        y_coords = sorted(y_coords)
        
        # Check if there are two valid horizontal or vertical cuts
        for i in range(1, len(x_coords) - 1):
            left_rectangles = []
            middle_rectangles = []
            right_rectangles = []
            
            for start_x, start_y, end_x, end_y in rectangles:
                if start_x < x_coords[i] < end_x:
                    middle_rectangles.append((start_x, start_y, end_x, end_y))
                elif start_x < x_coords[i]:
                    left_rectangles.append((start_x, start_y, end_x, end_y))
                else:
                    right_rectangles.append((start_x, start_y, end_x, end_y))
            
            if left_rectangles and middle_rectangles and right_rectangles:
                return True
        
        for i in range(1, len(y_coords) - 1):
            bottom_rectangles = []
            middle_rectangles = []
            top_rectangles = []
            
            for start_x, start_y, end_x, end_y in rectangles:
                if start_y < y_coords[i] < end_y:
                    middle_rectangles.append((start_x, start_y, end_x, end_y))
                elif start_y < y_coords[i]:
                    bottom_rectangles.append((start_x, start_y, end_x, end_y))
                else:
                    top_rectangles.append((start_x, start_y, end_x, end_y))
            
            if bottom_rectangles and middle_rectangles and top_rectangles:
                return True
        
        return False