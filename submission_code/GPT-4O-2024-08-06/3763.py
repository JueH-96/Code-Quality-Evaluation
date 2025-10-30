class Solution:
    def separateSquares(self, squares: List[List[int]]) -> float:
        def total_area_below(y):
            total_area = 0
            for x, y0, l in squares:
                if y0 < y:
                    h = min(l, y - y0)
                    total_area += l * h
            return total_area

        # Calculate the total area of all squares
        total_area = sum(l * l for _, _, l in squares)
        
        # We need to find the line y such that the area below it is half of the total area
        target_area = total_area / 2
        
        # Binary search for the y-coordinate
        low, high = 0, max(y + l for _, y, l in squares)
        while high - low > 1e-6:
            mid = (low + high) / 2
            if total_area_below(mid) < target_area:
                low = mid
            else:
                high = mid
        
        return low