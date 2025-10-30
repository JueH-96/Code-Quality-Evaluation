class Solution:
    def separateSquares(self, squares: List[List[int]]) -> float:
        def compute_area_below(y_candidate):
            total = 0.0
            for x, y, l in squares:
                if y_candidate <= y:
                    continue
                top = y + l
                if y_candidate >= top:
                    total += l * l
                else:
                    total += (y_candidate - y) * l
            return total
        
        total_area = 0.0
        min_y = float('inf')
        max_y = -float('inf')
        for x, y, l in squares:
            total_area += l * l
            min_y = min(min_y, y)
            max_y = max(max_y, y + l)
        
        half_area = total_area / 2.0
        
        left = min_y
        right = max_y
        answer = right
        
        while right - left > 1e-7:
            mid = (left + right) / 2.0
            area = compute_area_below(mid)
            if area >= half_area:
                answer = mid
                right = mid
            else:
                left = mid
        
        return answer