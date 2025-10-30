from typing import List

class Solution:
    def separateSquares(self, squares: List[List[int]]) -> float:
        """
        Finds the smallest y–coordinate of a horizontal line that splits the
        (possibly overlapping) squares into two equal total areas – one above
        the line and one below it.

        A square contributes its full area when the line is below its bottom,
        contributes 0 when the line is above its top, and contributes a
        rectangle of width `l` and height `y_i + l - h` when the line cuts
        through it.
        """
        # ------------------------------------------------------------------ #
        # Pre-processing: total area and search interval                      #
        # ------------------------------------------------------------------ #
        min_y = float('inf')
        max_top = 0
        total_area = 0.0                       # we keep the running sum as float

        for _, y, l in squares:
            total_area += l * l
            min_y = min(min_y, y)
            max_top = max(max_top, y + l)

        half_area = total_area / 2.0           # target area that must lie above

        # ------------------------------------------------------------------ #
        # Helper: area of all squares lying strictly above height `h`         #
        # ------------------------------------------------------------------ #
        def area_above(h: float) -> float:
            acc = 0.0
            for _, y, l in squares:
                if h <= y:                     # line below the square
                    acc += l * l
                elif h < y + l:                # line cuts the square
                    acc += l * (y + l - h)
                # else: line is above the square ⇒ contributes 0
            return acc

        # ------------------------------------------------------------------ #
        # Binary search for the root of f(h) = area_above(h) − half_area      #
        # area_above is continuous, strictly decreasing in h                  #
        # ------------------------------------------------------------------ #
        lo, hi = min_y, max_top               # f(lo) > 0, f(hi) < 0
        for _ in range(70):                   # 2⁻⁷⁰ ≈ 8e-22, easily < 1e-5 here
            mid = (lo + hi) / 2.0
            if area_above(mid) > half_area:
                lo = mid                      # still too much area above, move up
            else:
                hi = mid                      # area above is small enough, move down

        # `hi` is the smallest height giving area_above ≤ half_area
        return hi