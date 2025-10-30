class Solution:
    def separateSquares(self, squares: List[List[int]]) -> float:
        # Calculate total area
        total_area = sum(l * l for x, y, l in squares)
        target_area = total_area / 2
        
        # Binary search for the height
        lo = 0.0
        hi = max(y + l for x, y, l in squares)  # max possible height
        
        while hi - lo > 1e-12:  # Use higher precision than required
            mid = (lo + hi) / 2
            
            # Calculate area below the line at height mid
            area_below = 0
            for x, y, l in squares:
                if mid <= y:
                    # Entire square is above the line
                    pass
                elif mid >= y + l:
                    # Entire square is below the line
                    area_below += l * l
                else:
                    # Square is split
                    area_below += (mid - y) * l
            
            if area_below < target_area:
                lo = mid
            else:
                hi = mid
        
        return (lo + hi) / 2