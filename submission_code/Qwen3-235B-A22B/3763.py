from typing import List

class Solution:
    def separateSquares(self, squares: List[List[int]]) -> float:
        def get_area(c: float) -> float:
            area = 0.0
            for x, y, l in squares:
                if c >= y + l:
                    area += l * l
                elif c <= y:
                    continue
                else:
                    area += (c - y) * l
            return area
        
        total_area = 0.0
        min_y = float('inf')
        max_y_plus_l = -float('inf')
        
        for x, y, l in squares:
            total_area += l * l
            if y < min_y:
                min_y = y
            current = y + l
            if current > max_y_plus_l:
                max_y_plus_l = current
        
        target = total_area / 2.0
        low = min_y
        high = max_y_plus_l
        
        # Perform binary search for precision
        for _ in range(100):
            mid = (low + high) / 2
            cur_area = get_area(mid)
            if cur_area < target:
                low = mid
            else:
                high = mid
        
        return (low + high) / 2