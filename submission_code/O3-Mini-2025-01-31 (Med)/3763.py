from typing import List

class Solution:
    def separateSquares(self, squares: List[List[int]]) -> float:
        # Compute the total area. Note: each square's area is l^2.
        total_area = 0
        for x, y, l in squares:
            total_area += l * l
        half_area = total_area / 2.0

        # Define function to compute total area below the horizontal line at height H.
        def area_below(H: float) -> float:
            area = 0.0
            for x, y, l in squares:
                if H <= y:
                    # The whole square is above the line.
                    continue
                elif H >= y + l:
                    # The whole square is below the line.
                    area += l * l
                else:
                    # The line cuts this square.
                    # The vertical thickness below the line is H - y.
                    # Since the square's width is l, and the square is axis-aligned,
                    # the area below is (H - y) * l.
                    area += (H - y) * l
            return area

        # Determine search bounds.
        # Lower bound: the minimum bottom among all squares.
        # Upper bound: the maximum top among all squares.
        lo = min(y for _, y, _ in squares)
        hi = max(y + l for _, y, l in squares)
        
        # Binary search for the minimal H such that area_below(H) == half_area.
        # f(H) = total area below the line is monotonically non-decreasing with H.
        # We need to find the smallest H for which f(H) >= half_area.
        eps = 1e-7  # precision tolerance
        
        while hi - lo > eps:
            mid = (lo + hi) / 2.0
            if area_below(mid) < half_area:
                lo = mid
            else:
                hi = mid
        
        return hi