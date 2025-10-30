class Solution:
    def separateSquares(self, squares: List[List[int]]) -> float:
        # Calculate total area of all squares
        total_area = sum(l**2 for _, _, l in squares)
        
        # Target area below the line is half the total area
        target_area = total_area / 2
        
        # Define the range for binary search
        min_y = min(y for _, y, _ in squares)
        max_y = max(y + l for _, y, l in squares)
        
        # Binary search to find the y-coordinate
        left, right = min_y, max_y
        while right - left > 1e-5:  # Precision threshold from problem statement
            mid = (left + right) / 2
            area_below = self.calculate_area_below(squares, mid)
            
            if area_below < target_area:
                left = mid
            else:
                right = mid
                
        return left
    
    def calculate_area_below(self, squares, h):
        area_below = 0
        for x, y, l in squares:
            if h <= y:
                # The entire square is above the line
                continue
            elif h >= y + l:
                # The entire square is below the line
                area_below += l**2
            else:
                # The square is intersected by the line
                # Only count the portion below the line
                area_below += (h - y) * l
        return area_below