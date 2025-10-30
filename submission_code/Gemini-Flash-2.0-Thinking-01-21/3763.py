from typing import List

class Solution:
    def separateSquares(self, squares: List[List[int]]) -> float:
        # Calculate the total area of all squares
        total_area = 0
        for _, _, l in squares:
            # l * l can be up to (10^9)^2 = 10^18, fits in Python int
            total_area += l * l

        # The target area below the line is half of the total area
        # Use float for target area as division might result in fraction
        target_area_below = total_area / 2.0

        # Binary search range for the horizontal line y = h
        # Minimum possible y coordinate is 0.
        # Maximum possible y coordinate a square reaches is y_i + l_i.
        # Max y_i + l_i is 10^9 + 10^9 = 2e9.
        # The relevant height h is likely within [0, max(y_i + l_i)].
        # Set a sufficiently large upper bound.
        low = 0.0
        high = 2e9 + 7.0 # A safe upper bound slightly above the max possible y + l

        # Binary search iterations. 100 iterations provide high precision.
        # The interval width decreases by half in each step.
        # After N steps, width = (initial_width) / 2^N.
        # initial_width approx 2e9. Need error < 10^-5.
        # N approx log2((2e9) / 10^-5) = log2(2e14) approx 47.5.
        # 100 iterations is very safe for the required precision.
        for _ in range(100):
            mid = (low + high) / 2.0
            
            area_below_mid = 0.0
            for _, y, l in squares:
                y_f = float(y)
                l_f = float(l)
                
                # Calculate the height of the part of the square below the line mid
                # Square is [y, y + l] in y-range. Line is y = mid.
                # Part below is from y to min(mid, y + l).
                # Height is min(mid, y + l) - y, clamped at 0.
                # Using max(0.0, min(mid - y, l)) is numerically better.
                # height_below = max(0.0, min(mid, y_f + l_f) - y_f) # Equivalent, but might have precision issues
                height_below = max(0.0, min(mid - y_f, l_f)) # Numerically better way to calculate min(mid, y_f + l_f) - y_f

                # Area contribution of this square below line mid
                # Area = width * height = l * height_below
                area_below_mid += l_f * height_below

            # If the area below the line at 'mid' is less than the target,
            # the required line height must be above 'mid'.
            if area_below_mid < target_area_below:
                low = mid
            # If the area is equal to or greater than the target,
            # 'mid' is a possible candidate height, or the required height is lower.
            # We want the minimum such height, so we explore the lower half including mid.
            else:
                high = mid
        
        # After 100 iterations, 'high' will be very close to the minimum height
        # where the area below is >= target_area_below. Due to monotonicity,
        # this is the required height where AreaBelow(h) = TargetAreaBelow.
        return high