class Solution:
    def separateSquares(self, squares: List[List[int]]) -> float:
        def calculate_area(y_line: float) -> tuple:
            area_above = area_below = 0
            
            for x, y, l in squares:
                # Calculate intersection of square with the line
                square_top = y + l
                square_bottom = y
                
                if y_line <= square_bottom:
                    # Square is completely above the line
                    area_above += l * l
                elif y_line >= square_top:
                    # Square is completely below the line
                    area_below += l * l
                else:
                    # Square is intersected by the line
                    height_above = square_top - y_line
                    height_below = y_line - square_bottom
                    area_above += l * height_above
                    area_below += l * height_below
                    
            return area_above, area_below
        
        # Binary search for the y-coordinate
        left = float('inf')
        right = float('-inf')
        
        # Find the range of y-coordinates
        for x, y, l in squares:
            left = min(left, y)
            right = max(right, y + l)
        
        # Binary search with precision
        for _ in range(100):  # Sufficient iterations for required precision
            mid = (left + right) / 2
            area_above, area_below = calculate_area(mid)
            
            if abs(area_above - area_below) < 1e-6:
                return mid
            elif area_above > area_below:
                left = mid
            else:
                right = mid
                
        return left