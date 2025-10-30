from typing import List

class Solution:
    def separateSquares(self, squares: List[List[int]]) -> float:
        total_area = 0.0
        max_y_top = 0.0
        
        for x, y, l in squares:
            square_area = l * l
            total_area += square_area
            current_y_top = y + l
            if current_y_top > max_y_top:
                max_y_top = current_y_top
        
        if total_area == 0:
            return 0.0
        
        target = total_area / 2.0
        low = 0.0
        high = max_y_top
        
        # Perform binary search for a sufficient number of iterations to ensure precision
        for _ in range(100):
            mid = (low + high) / 2
            current_area = 0.0
            
            for x, y, l in squares:
                bottom = y
                top = y + l
                
                if mid <= bottom:
                    contrib = 0.0
                elif mid >= top:
                    contrib = l * l
                else:
                    contrib = l * (mid - bottom)
                current_area += contrib
            
            if current_area < target:
                low = mid
            else:
                high = mid
        
        return (low + high) / 2