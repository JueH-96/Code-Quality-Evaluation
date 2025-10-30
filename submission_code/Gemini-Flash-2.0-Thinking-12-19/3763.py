from typing import List

class Solution:
    def separateSquares(self, squares: List[List[int]]) -> float:
        # Calculate total area of all squares
        total_area = 0.0
        for _, _, l in squares:
            # Use float for area calculation to handle potentially large values and ensure precision
            total_area += float(l) * float(l)

        # The target area below the separating line is half of the total area
        target_area_below = total_area / 2.0

        # Define a function to calculate the total area of parts of squares below a given horizontal line y=h
        def calculate_area_below(h: float) -> float:
            current_total_area_below = 0.0
            for _, y, l in squares:
                y_float = float(y)
                l_float = float(l)
                
                # The height of the part of the square below the line y=h.
                # This is the height of the intersection of the square's y-range [y, y+l] with the range (-inf, h].
                # The intersection y-range is [y, min(y+l, h)].
                # The raw height is min(y+l, h) - y. We take max(0.0, ...) to handle cases where h <= y.
                height_below = max(0.0, min(h, y_float + l_float) - y_float)
                
                # The area contributed by this square below the line y=h is its width (l) multiplied by this height.
                current_total_area_below += l_float * height_below
            return current_total_area_below

        # Binary search for the height h.
        # The search range for h should cover all relevant y-coordinates.
        # The minimum possible y-coordinate is 0 (from constraint y_i >= 0).
        # The maximum possible relevant y-coordinate is the maximum top edge of any square (y_i + l_i).
        # Given constraints 0 <= y_i <= 10^9 and 1 <= l_i <= 10^9, the maximum top edge is 10^9 + 10^9 = 2e9.
        # A safe search range for h is [0.0, 2e9]. Values of h outside this range will result in AreaBelow(h) being either 0 or total_area.
        low = 0.0
        high = 2e9

        # Perform binary search for a fixed number of iterations to achieve the required precision (10^-5).
        # The initial range width is 2e9. To achieve precision 10^-5, we need range_width / 2^k <= 10^-5.
        # 2e9 / 2^k <= 10^-5  => 2e14 <= 2^k => k >= log2(2e14) ~= 47.5.
        # 100 iterations provide ample precision (range_width / 2^100 ~ 10^-21), far exceeding the requirement.
        num_iterations = 100
        for _ in range(num_iterations):
            mid = (low + high) / 2.0
            area_below_mid = calculate_area_below(mid)

            # If the calculated area below the midpoint height is less than the target area,
            # it means the splitting line needs to be higher to include more area below it.
            if area_below_mid < target_area_below:
                low = mid
            # If the calculated area is equal to or greater than the target,
            # the midpoint height is a potential answer, or the answer lies at a lower height.
            # We try to find the minimum such height.
            else:
                high = mid

        # After the specified number of iterations, `low` and `high` will have converged to a very small interval
        # containing the minimum height h where AreaBelow(h) >= target_area_below.
        # Either `low` or `high` can be returned as the result within the required precision.
        # Returning `high` represents the found value that meets or exceeds the target area from the lower bound search.
        return high