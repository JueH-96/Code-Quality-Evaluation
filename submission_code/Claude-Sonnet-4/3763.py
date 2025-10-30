class Solution:
    def separateSquares(self, squares: List[List[int]]) -> float:
        def getAreaDifference(h):
            # Returns area_above - area_below for horizontal line at height h
            area_above = 0
            area_below = 0
            
            for x, y, l in squares:
                square_area = l * l
                
                if h <= y:
                    # Entire square is above the line
                    area_above += square_area
                elif h >= y + l:
                    # Entire square is below the line
                    area_below += square_area
                else:
                    # Square is split by the line
                    # Area below = (h - y) * l
                    # Area above = (y + l - h) * l
                    area_below += (h - y) * l
                    area_above += (y + l - h) * l
            
            return area_above - area_below
        
        # Find the range of possible y-coordinates
        min_y = min(y for x, y, l in squares)
        max_y = max(y + l for x, y, l in squares)
        
        # Binary search for the y-coordinate where area difference is 0
        left, right = min_y, max_y
        
        # Binary search with sufficient precision
        for _ in range(100):  # 100 iterations gives us very high precision
            mid = (left + right) / 2
            diff = getAreaDifference(mid)
            
            if diff > 0:
                # More area above, need to move line up
                left = mid
            else:
                # More area below (or equal), need to move line down
                right = mid
        
        return (left + right) / 2