from typing import List

class Solution:
    def separateSquares(self, squares: List[List[int]]) -> float:
        # Compute total area
        total_area = 0.0
        min_y = float('inf')
        max_y = 0.0
        for x, y, l in squares:
            total_area += l * l
            min_y = min(min_y, y)
            max_y = max(max_y, y + l)
        
        target = total_area / 2.0
        
        # Function to compute area below horizontal line at height h
        def area_below(h: float) -> float:
            area = 0.0
            for x, y, l in squares:
                if h <= y:
                    # line is below the square
                    continue
                elif h >= y + l:
                    # line is above the square
                    area += l * l
                else:
                    # line cuts the square
                    height = h - y
                    area += l * height
            return area
        
        # Binary search for the y-coordinate where area_below(y) = target
        lo, hi = min_y, max_y
        eps = 1e-6
        while hi - lo > eps:
            mid = (lo + hi) / 2.0
            if area_below(mid) < target:
                lo = mid
            else:
                hi = mid
        
        return (lo + hi) / 2.0