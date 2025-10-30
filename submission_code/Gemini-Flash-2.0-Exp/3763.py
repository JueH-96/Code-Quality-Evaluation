class Solution:
    def separateSquares(self, squares: List[List[int]]) -> float:
        total_area = 0
        for x, y, l in squares:
            total_area += l * l
        
        low = min(y for x, y, l in squares)
        high = max(y + l for x, y, l in squares)
        
        while high - low > 1e-6:
            mid = (low + high) / 2
            
            area_below = 0
            for x, y, l in squares:
                if y >= mid:
                    continue
                
                if y + l <= mid:
                    area_below += l * l
                else:
                    area_below += (mid - y) * l
            
            if area_below * 2 < total_area:
                low = mid
            else:
                high = mid
        
        return low