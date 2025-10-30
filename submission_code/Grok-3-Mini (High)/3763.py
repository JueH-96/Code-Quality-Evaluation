from typing import List

class Solution:
    def separateSquares(self, squares: List[List[int]]) -> float:
        # Compute total area
        total_area = 0.0
        for sq in squares:
            l = sq[2]
            total_area += l * l
        target = total_area / 2.0
        
        # Find min y_bot and max y_top
        min_y = min(sq[1] for sq in squares)
        max_y = max(sq[1] + sq[2] for sq in squares)
        
        # Binary search for the smallest y such that area above <= target
        low = float(min_y)
        high = float(max_y)
        for _ in range(60):  # 60 iterations for precision
            mid = (low + high) / 2.0
            area_mid = 0.0
            for sq in squares:
                y_bot = sq[1]
                l = sq[2]
                y_top = y_bot + l
                if mid <= y_bot:
                    area_mid += l * l
                elif mid >= y_top:
                    pass  # area above is 0
                else:
                    area_mid += l * (y_top - mid)
            if area_mid > target:
                low = mid  # increase y to decrease area above
            else:
                high = mid  # decrease y or keep to find minimum y
        
        # Return the average of low and high as the approximate y
        return (low + high) / 2.0