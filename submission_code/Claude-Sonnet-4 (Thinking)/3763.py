class Solution:
    def separateSquares(self, squares: List[List[int]]) -> float:
        def area_below(y_line):
            total = 0
            for x, y, l in squares:
                if y_line <= y:
                    continue
                elif y_line >= y + l:
                    total += l * l
                else:
                    height_below = y_line - y
                    total += l * height_below
            return total
        
        total_area = sum(l * l for x, y, l in squares)
        target = total_area / 2
        
        min_y = min(y for x, y, l in squares)
        max_y = max(y + l for x, y, l in squares)
        
        left, right = min_y, max_y
        while right - left > 1e-9:
            mid = (left + right) / 2
            if area_below(mid) < target:
                left = mid
            else:
                right = mid
        
        return right