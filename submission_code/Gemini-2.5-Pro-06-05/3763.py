from typing import List

class Solution:
    def separateSquares(self, squares: List[List[int]]) -> float:
        """
        Finds the minimum y-coordinate of a horizontal line that splits the total area of squares equally.
        """
        
        # Calculate the total area of all squares. Overlapping areas are
        # counted multiple times as per the problem statement.
        total_area = 0
        for _, _, l in squares:
            total_area += l * l
        
        # The target area is half of the total area.
        target_area = total_area / 2.0

        # A helper function to calculate the total area of all squares below a line y=h.
        def get_area_below(h: float) -> float:
            area = 0.0
            for _, y, l in squares:
                if h > y:
                    # The line is above the bottom of the square.
                    # The height of the portion of the square below the line is the
                    # distance from the bottom edge (y) up to the line (h), but
                    # capped at the square's full height (l).
                    height_below = min(l, h - y)
                    area += l * height_below
            return area

        # We perform a binary search for the y-coordinate h.
        # The lower bound for h can be 0, as all y_i >= 0.
        low = 0.0
        # The upper bound for h can be the max possible y_i + l_i, which is
        # 10^9 + 10^9 = 2*10^9. This is a safe upper bound.
        high = 2 * 10**9

        # We iterate a fixed number of times (e.g., 100) to ensure sufficient precision.
        # This is more robust for floating-point binary search than using an epsilon.
        # 100 iterations on a range of 2*10^9 reduces the interval size to
        # (2*10^9) / 2^100, which is far smaller than the required 10^-5.
        for _ in range(100):
            mid = low + (high - low) / 2
            
            current_area = get_area_below(mid)
            
            if current_area < target_area:
                # If the area below mid is too small, we need to raise the line.
                # So, the solution must be in the upper half: [mid, high].
                low = mid
            else:
                # If the area is sufficient or too large, mid is a potential answer,
                # or the true answer is lower. Since we need the minimum h,
                # we search in the lower half: [low, mid].
                high = mid
        
        # After the loop, `high` will have converged to the minimum h that satisfies
        # the condition. `low` will be infinitesimally smaller than `high`.
        return high