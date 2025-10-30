class Solution:
    def separateSquares(self, squares: List[List[int]]) -> float:
        # Calculate total area
        total_area = sum(l * l for _, _, l in squares)
        target_area = total_area / 2
        
        # Binary search bounds
        left = 0
        right = max(y + l for _, y, l in squares)
        
        # Binary search for the y-coordinate
        while right - left > 1e-9:
            mid = (left + right) / 2
            
            # Calculate area above the line at height mid
            area_above = 0
            for x, y, l in squares:
                if mid <= y:
                    # Entire square is above
                    area_above += l * l
                elif mid < y + l:
                    # Partial square is above
                    area_above += (y + l - mid) ** 2
                # If mid >= y + l, square is entirely below (contributes 0)
            
            if area_above > target_area:
                left = mid
            else:
                right = mid
        
        return left