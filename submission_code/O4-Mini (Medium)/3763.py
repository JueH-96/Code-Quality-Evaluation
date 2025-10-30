from typing import List

class Solution:
    def separateSquares(self, squares: List[List[int]]) -> float:
        # Precompute for each square: bottom y, side length l, and top y = y + l
        ys = []
        ls = []
        ts = []
        total_area = 0.0
        low = float('inf')
        high = -float('inf')
        
        for x, y, l in squares:
            ys.append(y)
            ls.append(l)
            top = y + l
            ts.append(top)
            total_area += l * l
            low = min(low, y)
            high = max(high, top)
        
        # We want the horizontal line y = h such that
        # sum_i area_below_i(h) = total_area / 2
        target = total_area * 0.5
        
        left, right = low, high
        # Binary search on h
        for _ in range(60):
            mid = 0.5 * (left + right)
            below = 0.0
            # Compute area below the line y = mid
            for y, l, top in zip(ys, ls, ts):
                if mid <= y:
                    # line is below this square entirely
                    continue
                elif mid >= top:
                    # line is above this square entirely
                    below += l * l
                else:
                    # line cuts through this square
                    below += (mid - y) * l
            
            if below < target:
                left = mid
            else:
                right = mid
        
        return left