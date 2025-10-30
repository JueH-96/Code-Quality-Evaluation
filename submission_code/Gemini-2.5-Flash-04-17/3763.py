from typing import List

class Solution:
    def separateSquares(self, squares: List[List[int]]) -> float:
        
        # Helper function to calculate the total area of squares below a given y_line
        # Squares are represented as [x, y, l].
        # The area below y_line for a single square [x, y, l] is the area of the rectangle
        # [x, x+l] x [y, min(y+l, y_line)]. The height of this rectangle is max(0, min(y+l, y_line) - y).
        # The area is l * max(0, min(y+l, y_line) - y).
        def calculate_area_below(y_line: float, squares: List[List[int]]) -> float:
            area_below = 0.0
            for x, y, l in squares:
                # Convert integer coordinates and length to float for calculations with y_line
                y_float = float(y)
                l_float = float(l)
                y_top_float = y_float + l_float # y + l

                # The height of the portion of the square that is below the horizontal line y_line.
                # This is the length of the intersection of the square's y-range [y_float, y_top_float]
                # and the interval (-inf, y_line].
                # The intersection along the y-axis is the interval [y_float, min(y_top_float, y_line)].
                # The length of this interval is min(y_top_float, y_line) - y_float.
                # We take max(0.0, ...) because if y_line is less than or equal to y_float, the intersection
                # is empty or a single point, and the height below the line is 0.
                h_below = max(0.0, min(y_top_float, y_line) - y_float)

                # The area of this portion below the line is the width (l) multiplied by the height (h_below).
                area_below += l_float * h_below

            return area_below

        # Calculate the total area of all squares.
        # The problem guarantees that the total area sum(l_i * l_i) does not exceed 10^12.
        # This sum fits within standard double-precision floating-point type (float in Python).
        total_area = 0.0
        for x, y, l in squares:
            total_area += float(l) * float(l)

        # The target area for the region below the balancing line is exactly half of the total area.
        target_area = total_area / 2.0

        # We need to find the minimum y_line such that the total area of squares below it
        # equals the total area of squares above it. This is equivalent to finding the minimum y_line
        # such that the area below it equals half of the total area (A_below = A_above = TotalArea / 2).
        # The function `calculate_area_below(y_line)` is a monotonically non-decreasing function of y_line.
        # We can use binary search to find the y_line where `calculate_area_below(y_line)` equals `target_area`.

        # Determine a suitable search range for y_line.
        # The minimum possible y-coordinate for any part of any square is 0 (since y_i >= 0).
        # If the line is at or below the minimum y_i of all squares, the area below is 0.
        low = 0.0
        
        # The maximum relevant y-coordinate is the highest point reached by any square (max(y_i + l_i)).
        # If the line is at or above the maximum y_i + l_i of all squares, all squares are entirely below it,
        # and the area below equals the total area.
        # The balance line must be within the range [min(y_i), max(y_i + l_i)] if the total area is non-zero,
        # which is guaranteed by the constraint 1 <= l_i.
        max_y_plus_l = 0.0
        for x, y, l in squares:
             max_y_plus_l = max(max_y_plus_l, float(y) + float(l))
        high = max_y_plus_l

        # Perform binary search for a fixed number of iterations to achieve the required precision.
        # The problem requires an answer within 10^-5 of the actual answer.
        # The search range size is at most max_y_plus_l (around 2e9).
        # To achieve an absolute error Epsilon over a range R, we need N iterations such that R / 2^N < Epsilon.
        # For R=3e9 and Epsilon=1e-5, N > log2(3e9 / 1e-5) = log2(3e14) approx 48.06.
        # 100 iterations are more than sufficient for typical double precision and this requirement.
        num_iterations = 100 

        for _ in range(num_iterations):
            # Calculate the midpoint of the current search range.
            # Using low + (high - low) / 2.0 is numerically more stable than (low + high) / 2.0
            # when low and high are very large.
            mid = low + (high - low) / 2.0
            
            # Calculate the total area of squares below the current line `mid`.
            area_below_mid = calculate_area_below(mid, squares)

            # Compare the calculated area below the line with the target area.
            if area_below_mid < target_area:
                # The area below the line `mid` is less than half the total area.
                # This means the balancing line must be positioned higher to increase the area below.
                # The solution must be in the upper half (greater than `mid`).
                low = mid
            else:
                # The area below the line `mid` is greater than or equal to half the total area.
                # This means `mid` is a possible candidate for the balancing line, or the actual minimum
                # balancing line is lower than `mid`.
                # We are looking for the minimum y_line, so we explore the lower half including `mid`.
                high = mid

        # After the binary search loop finishes, the `high` variable (and `low` variable, as they will be very close)
        # will hold the minimum y-coordinate such that the area below the line is approximately equal to (or just crosses)
        # the target area. Due to the choice of updating `high = mid` when `area_below_mid >= target_area`,
        # `high` converges to the smallest value `y` where `calculate_area_below(y)` is >= `target_area`.
        # This is exactly the minimum y-coordinate we are looking for.
        return high