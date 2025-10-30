class Solution:
    def separateSquares(self, squares: List[List[int]]) -> float:
        total_area = 0.0
        min_y = float('inf')
        max_y = 0.0
        
        for x, y, l in squares:
            total_area += l * l
            if y < min_y:
                min_y = y
            current_max = y + l
            if current_max > max_y:
                max_y = current_max
        
        target = total_area / 2.0
        low = min_y
        high = max_y
        
        # Perform binary search with sufficient iterations for precision
        for _ in range(100):
            if high - low < 1e-7:
                break
            mid = (low + high) / 2
            area = 0.0
            for x, y_i, l in squares:
                y_bottom = y_i
                y_top = y_i + l
                if mid <= y_bottom:
                    continue  # contribution is 0
                elif mid >= y_top:
                    area += l * l
                else:
                    h = mid - y_bottom
                    area += l * h
            if area < target:
                low = mid
            else:
                high = mid
        
        return (low + high) / 2